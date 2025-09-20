"""
Attribute-based Explanations
============================

Provides explanations based on facial attribute predictions and differences.
Analyzes which attributes contribute to verification decisions and how
attribute predictions differ between faces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from .base import BaseExplainer, ExplanationResult, get_attribute_names


class AttributeExplainer(BaseExplainer):
    """
    Explains face recognition decisions through facial attributes
    
    Analyzes attribute predictions and their confidence scores to provide
    interpretable explanations for identity verification and recognition.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', threshold: float = 0.0):
        """
        Initialize attribute explainer
        
        Args:
            model: Face recognition model with attribute prediction heads
            device: Device to run on
            threshold: Threshold for binary attribute predictions
        """
        super().__init__(model, device)
        self.threshold = threshold
        self.attribute_names = get_attribute_names()
        
    def explain(self, 
                image: torch.Tensor,
                return_raw_logits: bool = False) -> Dict[str, Any]:
        """
        Generate attribute-based explanation for a single image
        
        Args:
            image: Input image tensor [1, 3, H, W]
            return_raw_logits: Whether to return raw logits
            
        Returns:
            Dictionary containing attribute predictions and explanations
        """
        self.model.eval()
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
        
        # Extract attribute predictions
        if 'attribute_logits' in outputs:
            attribute_logits = outputs['attribute_logits'].cpu().numpy()
            attribute_probs = torch.sigmoid(outputs['attribute_logits']).cpu().numpy()
        else:
            raise ValueError("Model does not output attribute predictions")
        
        # Convert to predictions and confidences
        predictions = (attribute_probs > 0.5).astype(int)
        confidences = np.abs(attribute_probs - 0.5) * 2  # Confidence as distance from 0.5
        
        # Create attribute dictionary
        attribute_dict = {}
        confidence_dict = {}
        
        for i, attr_name in enumerate(self.attribute_names):
            attribute_dict[attr_name] = predictions[0, i]
            confidence_dict[attr_name] = confidences[0, i]
        
        # Get top confident attributes
        top_positive = self._get_top_attributes(attribute_dict, confidence_dict, positive=True)
        top_negative = self._get_top_attributes(attribute_dict, confidence_dict, positive=False)
        
        result = {
            'attribute_predictions': attribute_dict,
            'attribute_confidences': confidence_dict,
            'top_positive_attributes': top_positive,
            'top_negative_attributes': top_negative,
            'attribute_summary': self._generate_attribute_summary(top_positive, top_negative)
        }
        
        if return_raw_logits:
            result['raw_logits'] = attribute_logits
            
        return result
    
    def explain_verification(self, 
                           image1: torch.Tensor, 
                           image2: torch.Tensor,
                           similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Explain face verification through attribute differences
        
        Args:
            image1: First face image
            image2: Second face image  
            similarity_threshold: Threshold for verification decision
            
        Returns:
            Comprehensive attribute-based verification explanation
        """
        # Get attribute explanations for both images
        attr1 = self.explain(image1)
        attr2 = self.explain(image2)
        
        # Compute embeddings and similarity
        with torch.no_grad():
            outputs1 = self.model(image1.to(self.device))
            outputs2 = self.model(image2.to(self.device))
            
            emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
            emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
        
        # Compute attribute differences
        attr_differences = self._compute_attribute_differences(
            attr1['attribute_predictions'], 
            attr2['attribute_predictions'],
            attr1['attribute_confidences'],
            attr2['attribute_confidences']
        )
        
        # Analyze attribute agreement/disagreement
        agreement_analysis = self._analyze_attribute_agreement(
            attr1['attribute_predictions'],
            attr2['attribute_predictions'],
            attr1['attribute_confidences'], 
            attr2['attribute_confidences']
        )
        
        # Generate verification explanation
        verification_explanation = self._generate_verification_explanation(
            similarity, similarity_threshold, attr_differences, agreement_analysis
        )
        
        return {
            'similarity': similarity,
            'is_match': similarity > similarity_threshold,
            'threshold': similarity_threshold,
            'image1_attributes': attr1,
            'image2_attributes': attr2,
            'attribute_differences': attr_differences,
            'agreement_analysis': agreement_analysis,
            'verification_explanation': verification_explanation
        }
    
    def _get_top_attributes(self, 
                           predictions: Dict[str, int],
                           confidences: Dict[str, float], 
                           positive: bool = True,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top confident attributes (positive or negative)"""
        filtered_attrs = []
        
        for attr_name, pred in predictions.items():
            confidence = confidences[attr_name]
            
            if positive and pred == 1:  # Positive attributes
                filtered_attrs.append((attr_name, confidence))
            elif not positive and pred == 0:  # Negative attributes
                filtered_attrs.append((attr_name, confidence))
        
        # Sort by confidence and return top k
        filtered_attrs.sort(key=lambda x: x[1], reverse=True)
        return filtered_attrs[:top_k]
    
    def _compute_attribute_differences(self, 
                                     attr1: Dict[str, int],
                                     attr2: Dict[str, int], 
                                     conf1: Dict[str, float],
                                     conf2: Dict[str, float]) -> Dict[str, Dict]:
        """Compute differences between attribute predictions"""
        differences = {}
        
        for attr_name in self.attribute_names:
            pred1 = attr1[attr_name]
            pred2 = attr2[attr_name]
            conf1_val = conf1[attr_name]
            conf2_val = conf2[attr_name]
            
            # Compute difference metrics
            prediction_diff = abs(pred1 - pred2)  # 0 = same, 1 = different
            confidence_diff = abs(conf1_val - conf2_val)
            avg_confidence = (conf1_val + conf2_val) / 2
            
            differences[attr_name] = {
                'prediction_difference': prediction_diff,
                'confidence_difference': confidence_diff,
                'average_confidence': avg_confidence,
                'image1_prediction': pred1,
                'image2_prediction': pred2,
                'image1_confidence': conf1_val,
                'image2_confidence': conf2_val
            }
        
        return differences
    
    def _analyze_attribute_agreement(self,
                                   attr1: Dict[str, int],
                                   attr2: Dict[str, int],
                                   conf1: Dict[str, float], 
                                   conf2: Dict[str, float]) -> Dict[str, Any]:
        """Analyze agreement/disagreement between attribute predictions"""
        agreements = []
        disagreements = []
        
        for attr_name in self.attribute_names:
            pred1 = attr1[attr_name]
            pred2 = attr2[attr_name]
            conf1_val = conf1[attr_name]
            conf2_val = conf2[attr_name]
            avg_conf = (conf1_val + conf2_val) / 2
            
            if pred1 == pred2:
                agreements.append((attr_name, avg_conf))
            else:
                disagreements.append((attr_name, avg_conf))
        
        # Sort by confidence
        agreements.sort(key=lambda x: x[1], reverse=True)
        disagreements.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_agreements': len(agreements),
            'total_disagreements': len(disagreements),
            'agreement_rate': len(agreements) / len(self.attribute_names),
            'top_agreements': agreements[:10],
            'top_disagreements': disagreements[:10],
            'avg_agreement_confidence': np.mean([conf for _, conf in agreements]) if agreements else 0.0,
            'avg_disagreement_confidence': np.mean([conf for _, conf in disagreements]) if disagreements else 0.0
        }
    
    def _generate_attribute_summary(self, 
                                  top_positive: List[Tuple[str, float]],
                                  top_negative: List[Tuple[str, float]]) -> str:
        """Generate textual summary of attributes"""
        summary_parts = []
        
        if top_positive:
            pos_attrs = [f"{attr.replace('_', ' ').lower()} ({conf:.2f})" 
                        for attr, conf in top_positive[:3]]
            summary_parts.append(f"Strong positive attributes: {', '.join(pos_attrs)}")
        
        if top_negative:
            neg_attrs = [f"no {attr.replace('_', ' ').lower()} ({conf:.2f})" 
                        for attr, conf in top_negative[:3]]
            summary_parts.append(f"Strong negative attributes: {', '.join(neg_attrs)}")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_verification_explanation(self,
                                         similarity: float,
                                         threshold: float, 
                                         attr_differences: Dict[str, Dict],
                                         agreement_analysis: Dict[str, Any]) -> str:
        """Generate textual explanation for verification decision"""
        is_match = similarity > threshold
        agreement_rate = agreement_analysis['agreement_rate']
        
        # Start with decision
        decision_text = f"{'Match' if is_match else 'No match'} (similarity: {similarity:.3f}, threshold: {threshold:.3f})"
        
        # Add attribute agreement information
        agreement_text = f"Attribute agreement: {agreement_rate:.1%} ({agreement_analysis['total_agreements']}/{len(self.attribute_names)} attributes match)"
        
        # Highlight key differences/agreements
        key_info = []
        
        if agreement_analysis['top_agreements']:
            top_agreement = agreement_analysis['top_agreements'][0]
            key_info.append(f"strongest agreement on '{top_agreement[0].replace('_', ' ').lower()}' (confidence: {top_agreement[1]:.2f})")
        
        if agreement_analysis['top_disagreements']:
            top_disagreement = agreement_analysis['top_disagreements'][0]
            key_info.append(f"strongest disagreement on '{top_disagreement[0].replace('_', ' ').lower()}' (confidence: {top_disagreement[1]:.2f})")
        
        explanation_parts = [decision_text, agreement_text]
        if key_info:
            explanation_parts.append("Key factors: " + ", ".join(key_info))
        
        return ". ".join(explanation_parts) + "."
    
    def get_attribute_importance_for_identity(self,
                                            images: torch.Tensor,
                                            identities: torch.Tensor) -> Dict[str, float]:
        """
        Compute attribute importance for identity prediction
        
        Args:
            images: Batch of images [N, 3, H, W]
            identities: Corresponding identity labels [N]
            
        Returns:
            Dictionary of attribute importance scores
        """
        self.model.eval()
        images = images.to(self.device)
        identities = identities.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(images)
            
        identity_logits = outputs.get('identity_logits', outputs.get('embeddings'))
        attribute_logits = outputs['attribute_logits']
        
        # Compute attribute importance via gradient analysis
        importance_scores = {}
        
        for i, attr_name in enumerate(self.attribute_names):
            # Enable gradients for this attribute
            attr_logit = attribute_logits[:, i].requires_grad_(True)
            
            # Compute identity loss with this attribute
            identity_loss = F.cross_entropy(identity_logits, identities)
            
            # Compute gradient
            grad = torch.autograd.grad(identity_loss, attr_logit, 
                                     create_graph=False, retain_graph=True)[0]
            
            # Importance as gradient magnitude
            importance = grad.abs().mean().item()
            importance_scores[attr_name] = importance
        
        return importance_scores