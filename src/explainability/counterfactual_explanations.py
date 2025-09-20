"""
Counterfactual Explanations
===========================

Implements counterfactual analysis by synthetically manipulating attributes
and measuring how these changes affect similarity scores and verification decisions.
Helps identify which attributes are most important for model decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
from .base import BaseExplainer, ExplanationResult, get_attribute_names, tensor_to_image


class CounterfactualExplainer(BaseExplainer):
    """
    Counterfactual explanations through attribute manipulation
    
    Analyzes how changes to facial attributes affect model predictions
    to understand attribute sensitivity and importance.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda'):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Face recognition model with attribute prediction
            device: Device to run on
        """
        super().__init__(model, device)
        self.attribute_names = get_attribute_names()
        
    def analyze_attribute_sensitivity(self,
                                    image: torch.Tensor,
                                    target_attributes: Optional[List[str]] = None,
                                    perturbation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Analyze sensitivity to attribute changes via input perturbations
        
        Args:
            image: Input image [1, 3, H, W]
            target_attributes: Specific attributes to analyze
            perturbation_strength: Strength of input perturbations
            
        Returns:
            Attribute sensitivity analysis results
        """
        if target_attributes is None:
            target_attributes = self.attribute_names
        
        self.model.eval()
        image = image.to(self.device)
        
        # Get baseline predictions
        with torch.no_grad():
            baseline_outputs = self.model(image)
            baseline_embedding = baseline_outputs.get('embeddings', baseline_outputs.get('identity_logits'))
            baseline_attributes = baseline_outputs.get('attribute_logits')
        
        sensitivity_results = {}
        
        for attr_name in target_attributes:
            attr_idx = self.attribute_names.index(attr_name)
            
            # Create perturbations that should affect this attribute
            perturbations = self._create_attribute_perturbations(
                image, attr_idx, perturbation_strength
            )
            
            sensitivity_scores = []
            
            for perturbation in perturbations:
                with torch.no_grad():
                    perturbed_outputs = self.model(perturbation)
                    perturbed_embedding = perturbed_outputs.get('embeddings', perturbed_outputs.get('identity_logits'))
                    perturbed_attributes = perturbed_outputs.get('attribute_logits')
                
                # Compute changes
                embedding_change = torch.cosine_similarity(
                    baseline_embedding, perturbed_embedding, dim=1
                ).item()
                
                if baseline_attributes is not None and perturbed_attributes is not None:
                    attr_change = torch.abs(
                        baseline_attributes[0, attr_idx] - perturbed_attributes[0, attr_idx]
                    ).item()
                else:
                    attr_change = 0.0
                
                sensitivity_scores.append({
                    'embedding_similarity': embedding_change,
                    'attribute_change': attr_change
                })
            
            # Aggregate sensitivity
            avg_embedding_change = np.mean([s['embedding_similarity'] for s in sensitivity_scores])
            avg_attribute_change = np.mean([s['attribute_change'] for s in sensitivity_scores])
            
            sensitivity_results[attr_name] = {
                'embedding_sensitivity': 1.0 - avg_embedding_change,  # Lower similarity = higher sensitivity
                'attribute_sensitivity': avg_attribute_change,
                'individual_scores': sensitivity_scores
            }
        
        return {
            'sensitivity_scores': sensitivity_results,
            'top_sensitive_attributes': self._get_top_sensitive_attributes(sensitivity_results),
            'sensitivity_summary': self._generate_sensitivity_summary(sensitivity_results)
        }
    
    def counterfactual_verification_analysis(self,
                                           image1: torch.Tensor,
                                           image2: torch.Tensor,
                                           target_attributes: Optional[List[str]] = None,
                                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze how attribute changes affect verification decisions
        
        Args:
            image1: First face image
            image2: Second face image
            target_attributes: Attributes to analyze
            threshold: Verification threshold
            
        Returns:
            Counterfactual verification analysis
        """
        if target_attributes is None:
            # Focus on most discriminative attributes
            target_attributes = [
                'Male', 'Young', 'Eyeglasses', 'Smiling', 'Heavy_Makeup',
                'Mustache', 'Goatee', 'Bald', 'Wearing_Hat'
            ]
        
        # Get baseline verification result
        baseline_result = self._compute_verification_baseline(image1, image2, threshold)
        
        counterfactual_results = {}
        
        for attr_name in target_attributes:
            # Test adding/removing this attribute from both images
            cf_results = self._test_attribute_counterfactuals(
                image1, image2, attr_name, threshold
            )
            counterfactual_results[attr_name] = cf_results
        
        # Analyze which attributes cause decision flips
        decision_flip_analysis = self._analyze_decision_flips(
            baseline_result, counterfactual_results
        )
        
        return {
            'baseline_result': baseline_result,
            'counterfactual_results': counterfactual_results,
            'decision_flip_analysis': decision_flip_analysis,
            'attribute_importance_ranking': self._rank_attribute_importance(counterfactual_results)
        }
    
    def _create_attribute_perturbations(self,
                                      image: torch.Tensor,
                                      attr_idx: int,
                                      strength: float) -> List[torch.Tensor]:
        """Create input perturbations targeting specific attribute"""
        perturbations = []
        
        # Random noise perturbations
        for _ in range(5):
            noise = torch.randn_like(image) * strength
            perturbation = torch.clamp(image + noise, 0, 1)
            perturbations.append(perturbation)
        
        # Attribute-specific perturbations (simplified)
        attr_name = self.attribute_names[attr_idx]
        
        if 'Eyeglasses' in attr_name:
            # Add noise around eye regions (simplified approach)
            eye_perturbation = image.clone()
            eye_region = eye_perturbation[:, :, 80:140, 60:164]  # Approximate eye region
            eye_region += torch.randn_like(eye_region) * strength * 2
            perturbations.append(torch.clamp(eye_perturbation, 0, 1))
        
        elif 'Smiling' in attr_name:
            # Add noise around mouth region
            mouth_perturbation = image.clone()
            mouth_region = mouth_perturbation[:, :, 140:200, 80:144]  # Approximate mouth region
            mouth_region += torch.randn_like(mouth_region) * strength * 2
            perturbations.append(torch.clamp(mouth_perturbation, 0, 1))
        
        elif 'Male' in attr_name:
            # Add general facial structure noise
            face_perturbation = image.clone()
            face_perturbation += torch.randn_like(face_perturbation) * strength * 1.5
            perturbations.append(torch.clamp(face_perturbation, 0, 1))
        
        return perturbations
    
    def _compute_verification_baseline(self,
                                     image1: torch.Tensor,
                                     image2: torch.Tensor,
                                     threshold: float) -> Dict[str, Any]:
        """Compute baseline verification result"""
        with torch.no_grad():
            outputs1 = self.model(image1.to(self.device))
            outputs2 = self.model(image2.to(self.device))
            
            emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
            emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
            is_match = similarity > threshold
            
            # Get attribute predictions
            attr1 = outputs1.get('attribute_logits')
            attr2 = outputs2.get('attribute_logits')
            
            attribute_diffs = {}
            if attr1 is not None and attr2 is not None:
                for i, attr_name in enumerate(self.attribute_names):
                    diff = torch.abs(attr1[0, i] - attr2[0, i]).item()
                    attribute_diffs[attr_name] = diff
        
        return {
            'similarity': similarity,
            'is_match': is_match,
            'threshold': threshold,
            'attribute_differences': attribute_diffs
        }
    
    def _test_attribute_counterfactuals(self,
                                      image1: torch.Tensor,
                                      image2: torch.Tensor,
                                      attr_name: str,
                                      threshold: float) -> Dict[str, Any]:
        """Test counterfactuals for specific attribute"""
        attr_idx = self.attribute_names.index(attr_name)
        
        # Create synthetic modifications (simplified approach)
        modified_results = []
        
        # Test different perturbation strengths
        for strength in [0.05, 0.1, 0.2]:
            # Modify image1
            perturbations1 = self._create_attribute_perturbations(image1, attr_idx, strength)
            
            for pert1 in perturbations1[:2]:  # Limit for efficiency
                with torch.no_grad():
                    outputs1 = self.model(pert1.to(self.device))
                    outputs2 = self.model(image2.to(self.device))
                    
                    emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
                    emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
                    
                    similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
                    is_match = similarity > threshold
                    
                    # Check attribute change
                    attr_logits1 = outputs1.get('attribute_logits')
                    if attr_logits1 is not None:
                        attr_pred = torch.sigmoid(attr_logits1[0, attr_idx]).item()
                    else:
                        attr_pred = 0.5
                
                modified_results.append({
                    'modification_strength': strength,
                    'similarity': similarity,
                    'is_match': is_match,
                    'attribute_prediction': attr_pred,
                    'modified_image': 1  # Image 1 was modified
                })
            
            # Modify image2
            perturbations2 = self._create_attribute_perturbations(image2, attr_idx, strength)
            
            for pert2 in perturbations2[:2]:
                with torch.no_grad():
                    outputs1 = self.model(image1.to(self.device))
                    outputs2 = self.model(pert2.to(self.device))
                    
                    emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
                    emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
                    
                    similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
                    is_match = similarity > threshold
                    
                    attr_logits2 = outputs2.get('attribute_logits')
                    if attr_logits2 is not None:
                        attr_pred = torch.sigmoid(attr_logits2[0, attr_idx]).item()
                    else:
                        attr_pred = 0.5
                
                modified_results.append({
                    'modification_strength': strength,
                    'similarity': similarity,
                    'is_match': is_match,
                    'attribute_prediction': attr_pred,
                    'modified_image': 2  # Image 2 was modified
                })
        
        return {
            'attribute': attr_name,
            'modifications': modified_results,
            'average_similarity_change': np.mean([r['similarity'] for r in modified_results])
        }
    
    def _analyze_decision_flips(self,
                              baseline: Dict[str, Any],
                              counterfactuals: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze which attributes cause verification decision flips"""
        baseline_decision = baseline['is_match']
        decision_flips = {}
        
        for attr_name, cf_result in counterfactuals.items():
            flip_count = 0
            total_tests = len(cf_result['modifications'])
            
            for mod in cf_result['modifications']:
                if mod['is_match'] != baseline_decision:
                    flip_count += 1
            
            flip_rate = flip_count / total_tests if total_tests > 0 else 0.0
            decision_flips[attr_name] = {
                'flip_rate': flip_rate,
                'flip_count': flip_count,
                'total_tests': total_tests
            }
        
        # Sort by flip rate
        sorted_flips = sorted(decision_flips.items(), key=lambda x: x[1]['flip_rate'], reverse=True)
        
        return {
            'decision_flips': decision_flips,
            'most_influential_attributes': sorted_flips[:5],
            'flip_summary': self._generate_flip_summary(sorted_flips)
        }
    
    def _rank_attribute_importance(self, counterfactuals: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """Rank attributes by their importance for verification decisions"""
        importance_scores = []
        
        for attr_name, cf_result in counterfactuals.items():
            similarities = [mod['similarity'] for mod in cf_result['modifications']]
            
            if similarities:
                # Importance as variance in similarity scores
                importance = np.var(similarities)
                importance_scores.append((attr_name, importance))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores
    
    def _get_top_sensitive_attributes(self, sensitivity_results: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """Get attributes ranked by sensitivity"""
        sensitivities = []
        
        for attr_name, result in sensitivity_results.items():
            # Combined sensitivity score
            embedding_sens = result['embedding_sensitivity']
            attr_sens = result['attribute_sensitivity']
            combined_sens = (embedding_sens + attr_sens) / 2
            
            sensitivities.append((attr_name, combined_sens))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        return sensitivities[:10]
    
    def _generate_sensitivity_summary(self, sensitivity_results: Dict[str, Dict]) -> str:
        """Generate textual summary of sensitivity analysis"""
        top_sensitive = self._get_top_sensitive_attributes(sensitivity_results)
        
        if not top_sensitive:
            return "No sensitivity analysis results available."
        
        summary_parts = []
        for attr_name, sensitivity in top_sensitive[:3]:
            attr_clean = attr_name.replace('_', ' ').lower()
            sensitivity_level = "high" if sensitivity > 0.5 else "moderate" if sensitivity > 0.2 else "low"
            summary_parts.append(f"{attr_clean} ({sensitivity_level}: {sensitivity:.3f})")
        
        return f"Most sensitive attributes: {', '.join(summary_parts)}"
    
    def _generate_flip_summary(self, sorted_flips: List[Tuple[str, Dict]]) -> str:
        """Generate summary of decision flip analysis"""
        if not sorted_flips:
            return "No decision flips observed."
        
        summary_parts = []
        for attr_name, flip_info in sorted_flips[:3]:
            flip_rate = flip_info['flip_rate']
            attr_clean = attr_name.replace('_', ' ').lower()
            
            if flip_rate > 0.3:
                level = "frequently"
            elif flip_rate > 0.1:
                level = "occasionally" 
            else:
                level = "rarely"
            
            summary_parts.append(f"{attr_clean} {level} causes decision changes ({flip_rate:.1%})")
        
        return f"Decision influence: {', '.join(summary_parts)}"
    
    def explain(self, 
                image: torch.Tensor,
                task_type: str = 'sensitivity',
                **kwargs) -> Dict[str, Any]:
        """
        Generate counterfactual explanation
        
        Args:
            image: Input image tensor
            task_type: 'sensitivity' or 'verification'
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing counterfactual explanations
        """
        if task_type == 'sensitivity':
            return self.analyze_attribute_sensitivity(image, **kwargs)
        elif task_type == 'verification':
            if 'image2' not in kwargs:
                raise ValueError("Second image required for verification counterfactual analysis")
            return self.counterfactual_verification_analysis(image, kwargs['image2'], **kwargs)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def visualize_attribute_sensitivity(self,
                                      sensitivity_results: Dict[str, Dict],
                                      top_k: int = 10) -> Dict[str, Any]:
        """
        Create visualization of attribute sensitivity results
        
        Args:
            sensitivity_results: Results from analyze_attribute_sensitivity
            top_k: Number of top attributes to visualize
            
        Returns:
            Visualization data and summary
        """
        top_sensitive = self._get_top_sensitive_attributes(sensitivity_results)[:top_k]
        
        # Prepare data for visualization
        attr_names = [attr.replace('_', ' ') for attr, _ in top_sensitive]
        sensitivities = [sens for _, sens in top_sensitive]
        
        visualization_data = {
            'attribute_names': attr_names,
            'sensitivity_scores': sensitivities,
            'visualization_type': 'bar_chart',
            'title': 'Attribute Sensitivity Analysis',
            'xlabel': 'Facial Attributes',
            'ylabel': 'Sensitivity Score'
        }
        
        return visualization_data