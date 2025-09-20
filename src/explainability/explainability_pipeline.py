"""
Explainability Pipeline
======================

Unified interface that combines all explanation methods into comprehensive outputs.
Provides a single entry point for generating complete explanations for face
recognition and verification decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from .base import BaseExplainer, ExplanationResult
from .visual_explanations import GradCAMExplainer, IntegratedGradientsExplainer, GuidedGradCAMExplainer
from .attribute_explanations import AttributeExplainer
from .concept_analysis import TCAVAnalyzer
from .prototype_explanations import PrototypeExplainer
from .counterfactual_explanations import CounterfactualExplainer
from .textual_explanations import TextualExplainer


class ExplainabilityPipeline:
    """
    Comprehensive explainability pipeline for face recognition
    
    Integrates all explanation methods to provide complete interpretability
    for face recognition and verification decisions.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 explanation_methods: Optional[List[str]] = None,
                 explanation_style: str = 'comprehensive'):
        """
        Initialize explainability pipeline
        
        Args:
            model: Face recognition model
            device: Device to run on
            explanation_methods: List of methods to include ['gradcam', 'integrated_gradients', 
                               'attributes', 'tcav', 'prototypes', 'counterfactual']
            explanation_style: 'brief', 'comprehensive', or 'technical'
        """
        self.model = model
        self.device = device
        self.explanation_style = explanation_style
        
        if explanation_methods is None:
            explanation_methods = ['gradcam', 'attributes', 'prototypes', 'textual']
        
        self.enabled_methods = explanation_methods
        
        # Initialize explainers
        self.explainers = {}
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize all requested explainers"""
        if 'gradcam' in self.enabled_methods:
            self.explainers['gradcam'] = GradCAMExplainer(self.model, device=self.device)
        
        if 'guided_gradcam' in self.enabled_methods:
            self.explainers['guided_gradcam'] = GuidedGradCAMExplainer(self.model, device=self.device)
        
        if 'integrated_gradients' in self.enabled_methods:
            self.explainers['integrated_gradients'] = IntegratedGradientsExplainer(self.model, device=self.device)
        
        if 'attributes' in self.enabled_methods:
            self.explainers['attributes'] = AttributeExplainer(self.model, device=self.device)
        
        if 'tcav' in self.enabled_methods:
            self.explainers['tcav'] = TCAVAnalyzer(self.model, device=self.device)
        
        if 'prototypes' in self.enabled_methods:
            self.explainers['prototypes'] = PrototypeExplainer(self.model, device=self.device)
        
        if 'counterfactual' in self.enabled_methods:
            self.explainers['counterfactual'] = CounterfactualExplainer(self.model, device=self.device)
        
        if 'textual' in self.enabled_methods:
            self.explainers['textual'] = TextualExplainer(self.model, device=self.device, 
                                                         explanation_style=self.explanation_style)
    
    def explain_verification(self,
                           image1: torch.Tensor,
                           image2: torch.Tensor,
                           threshold: float = 0.5,
                           include_visualizations: bool = True) -> ExplanationResult:
        """
        Generate comprehensive explanation for face verification
        
        Args:
            image1: First face image [1, 3, H, W]
            image2: Second face image [1, 3, H, W]
            threshold: Verification threshold
            include_visualizations: Whether to generate visualization images
            
        Returns:
            Complete explanation result
        """
        print("Generating comprehensive verification explanation...")
        
        # Get basic model outputs
        with torch.no_grad():
            outputs1 = self.model(image1.to(self.device))
            outputs2 = self.model(image2.to(self.device))
            
            emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
            emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
            is_match = similarity > threshold
        
        # Initialize result
        result = ExplanationResult(
            embedding=emb1.cpu().numpy().flatten(),
            match_score=similarity
        )
        
        # Run each explanation method
        explanation_results = {}
        
        # Visual explanations
        if 'gradcam' in self.explainers:
            print("Generating Grad-CAM explanations...")
            gradcam_result = self.explainers['gradcam'].explain_verification(image1, image2, threshold)
            explanation_results['gradcam'] = gradcam_result
            result.grad_cam = gradcam_result['image1_gradcam']['grad_cam']
        
        if 'integrated_gradients' in self.explainers:
            print("Generating Integrated Gradients explanations...")
            ig_result1 = self.explainers['integrated_gradients'].explain(image1, target_type='identity')
            ig_result2 = self.explainers['integrated_gradients'].explain(image2, target_type='identity')
            explanation_results['integrated_gradients'] = {'image1': ig_result1, 'image2': ig_result2}
            result.integrated_gradients = ig_result1['integrated_gradients']
        
        # Attribute explanations
        if 'attributes' in self.explainers:
            print("Generating attribute explanations...")
            attr_result = self.explainers['attributes'].explain_verification(image1, image2, threshold)
            explanation_results['attributes'] = attr_result
            result.predicted_attributes = attr_result['image1_attributes']['attribute_predictions']
            result.attribute_confidences = attr_result['image1_attributes']['attribute_confidences']
            result.attribute_differences = attr_result['attribute_differences']
        
        # Prototype explanations
        if 'prototypes' in self.explainers:
            print("Generating prototype explanations...")
            # Note: Requires prototype database to be built first
            try:
                proto_result = self.explainers['prototypes'].explain_verification_with_prototypes(
                    image1, image2, threshold
                )
                explanation_results['prototypes'] = proto_result
                result.similar_prototypes = proto_result.get('similar_to_pair', [])
            except ValueError as e:
                print(f"Prototype explanation skipped: {e}")
                explanation_results['prototypes'] = None
        
        # Concept analysis (TCAV)
        if 'tcav' in self.explainers:
            print("Generating concept analysis...")
            try:
                concept_result1 = self.explainers['tcav'].explain(image1)
                concept_result2 = self.explainers['tcav'].explain(image2)
                explanation_results['tcav'] = {'image1': concept_result1, 'image2': concept_result2}
                result.concept_scores = concept_result1.get('tcav_scores', {})
            except Exception as e:
                print(f"TCAV analysis skipped: {e}")
                explanation_results['tcav'] = None
        
        # Counterfactual analysis
        if 'counterfactual' in self.explainers:
            print("Generating counterfactual analysis...")
            cf_result = self.explainers['counterfactual'].counterfactual_verification_analysis(
                image1, image2, threshold=threshold
            )
            explanation_results['counterfactual'] = cf_result
            result.counterfactual_changes = cf_result.get('decision_flip_analysis', {})
        
        # Textual explanation
        if 'textual' in self.explainers:
            print("Generating textual explanation...")
            textual_result = self.explainers['textual'].explain_verification(
                image1, image2, threshold,
                visual_explanation=explanation_results.get('gradcam'),
                attribute_explanation=explanation_results.get('attributes'),
                prototype_explanation=explanation_results.get('prototypes'),
                concept_explanation=explanation_results.get('tcav')
            )
            explanation_results['textual'] = textual_result
            result.textual_explanation = textual_result['explanation']
            result.explanation_confidence = textual_result['confidence']
        
        # Generate visualizations if requested
        if include_visualizations:
            visualizations = self._create_verification_visualizations(
                image1, image2, result, explanation_results
            )
            explanation_results['visualizations'] = visualizations
        
        # Store all results
        result.__dict__.update({'detailed_results': explanation_results})
        
        print("Verification explanation complete.")
        return result
    
    def explain_identity(self,
                        image: torch.Tensor,
                        predicted_identity: Optional[int] = None,
                        include_visualizations: bool = True) -> ExplanationResult:
        """
        Generate comprehensive explanation for identity prediction
        
        Args:
            image: Input face image [1, 3, H, W]
            predicted_identity: Known identity (if available)
            include_visualizations: Whether to generate visualization images
            
        Returns:
            Complete explanation result
        """
        print("Generating comprehensive identity explanation...")
        
        # Get basic model outputs
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            embedding = outputs.get('embeddings', outputs.get('identity_logits'))
            
            if 'identity_logits' in outputs and predicted_identity is None:
                predicted_identity = outputs['identity_logits'].argmax(dim=1).item()
        
        # Initialize result
        result = ExplanationResult(
            embedding=embedding.cpu().numpy().flatten(),
            identity_logits=outputs.get('identity_logits', torch.tensor([0.0])).cpu().numpy(),
            attribute_logits=outputs.get('attribute_logits', torch.tensor([0.0])).cpu().numpy()
        )
        
        # Run each explanation method
        explanation_results = {}
        
        # Visual explanations
        if 'gradcam' in self.explainers:
            print("Generating Grad-CAM explanation...")
            gradcam_result = self.explainers['gradcam'].explain(
                image, target_type='identity', target_class=predicted_identity
            )
            explanation_results['gradcam'] = gradcam_result
            result.grad_cam = gradcam_result['grad_cam']
        
        # Attribute explanations
        if 'attributes' in self.explainers:
            print("Generating attribute explanation...")
            attr_result = self.explainers['attributes'].explain(image)
            explanation_results['attributes'] = attr_result
            result.predicted_attributes = attr_result['attribute_predictions']
            result.attribute_confidences = attr_result['attribute_confidences']
        
        # Prototype explanations
        if 'prototypes' in self.explainers:
            print("Generating prototype explanation...")
            try:
                proto_result = self.explainers['prototypes'].explain_identity_with_prototypes(
                    image, predicted_identity
                )
                explanation_results['prototypes'] = proto_result
                result.similar_prototypes = proto_result.get('similar_prototypes', [])
                result.dissimilar_prototypes = proto_result.get('dissimilar_prototypes', [])
            except ValueError as e:
                print(f"Prototype explanation skipped: {e}")
                explanation_results['prototypes'] = None
        
        # Textual explanation
        if 'textual' in self.explainers:
            print("Generating textual explanation...")
            textual_result = self.explainers['textual'].explain_identity(
                image, predicted_identity,
                attribute_explanation=explanation_results.get('attributes'),
                visual_explanation=explanation_results.get('gradcam'),
                prototype_explanation=explanation_results.get('prototypes')
            )
            explanation_results['textual'] = textual_result
            result.textual_explanation = textual_result['explanation']
            result.explanation_confidence = textual_result['confidence']
        
        # Generate visualizations if requested
        if include_visualizations:
            visualizations = self._create_identity_visualizations(
                image, result, explanation_results
            )
            explanation_results['visualizations'] = visualizations
        
        # Store all results
        result.__dict__.update({'detailed_results': explanation_results})
        
        print("Identity explanation complete.")
        return result
    
    def _create_verification_visualizations(self,
                                          image1: torch.Tensor,
                                          image2: torch.Tensor,
                                          result: ExplanationResult,
                                          explanation_results: Dict) -> Dict[str, np.ndarray]:
        """Create visualization images for verification explanation"""
        visualizations = {}
        
        # Convert images to numpy
        img1_np = self._tensor_to_display_image(image1)
        img2_np = self._tensor_to_display_image(image2)
        
        # Create main comparison
        comparison = np.hstack([img1_np, img2_np])
        visualizations['image_comparison'] = comparison
        
        # Grad-CAM visualization
        if 'gradcam' in explanation_results and explanation_results['gradcam']:
            gradcam_data = explanation_results['gradcam']
            
            if 'image1_gradcam' in gradcam_data and 'image2_gradcam' in gradcam_data:
                gradcam1 = gradcam_data['image1_gradcam'].get('grad_cam_overlay', img1_np)
                gradcam2 = gradcam_data['image2_gradcam'].get('grad_cam_overlay', img2_np)
                
                gradcam_comparison = np.hstack([gradcam1, gradcam2])
                visualizations['gradcam_comparison'] = gradcam_comparison
        
        # Attribute comparison chart
        if 'attributes' in explanation_results and explanation_results['attributes']:
            attr_viz = self._create_attribute_comparison_chart(explanation_results['attributes'])
            visualizations['attribute_comparison'] = attr_viz
        
        return visualizations
    
    def _create_identity_visualizations(self,
                                      image: torch.Tensor,
                                      result: ExplanationResult,
                                      explanation_results: Dict) -> Dict[str, np.ndarray]:
        """Create visualization images for identity explanation"""
        visualizations = {}
        
        # Original image
        img_np = self._tensor_to_display_image(image)
        visualizations['original_image'] = img_np
        
        # Grad-CAM overlay
        if result.grad_cam is not None:
            # Create overlay using utility function
            grad_cam_overlay = self._overlay_saliency(img_np, result.grad_cam)
            visualizations['gradcam_overlay'] = grad_cam_overlay
        
        # Attribute chart
        if result.predicted_attributes is not None:
            attr_viz = self._create_attribute_chart(result.predicted_attributes, result.attribute_confidences)
            visualizations['attribute_chart'] = attr_viz
        
        # Prototype comparison
        if result.similar_prototypes:
            proto_viz = self._create_prototype_visualization(result.similar_prototypes)
            visualizations['prototype_comparison'] = proto_viz
        
        return visualizations
    
    def _create_attribute_comparison_chart(self, attr_result: Dict) -> np.ndarray:
        """Create attribute comparison chart for verification"""
        if 'attribute_differences' not in attr_result:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Get top differing attributes
        attr_diffs = attr_result['attribute_differences']
        top_diffs = sorted(attr_diffs.items(), 
                          key=lambda x: x[1]['prediction_difference'], reverse=True)[:10]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        attr_names = [attr.replace('_', ' ') for attr, _ in top_diffs]
        diff_values = [data['prediction_difference'] for _, data in top_diffs]
        
        ax.barh(attr_names, diff_values)
        ax.set_xlabel('Attribute Difference')
        ax.set_title('Top Attribute Differences Between Images')
        
        # Convert to image
        fig.canvas.draw()
        # Use buffer_rgba for newer matplotlib versions, fallback to tostring_rgb
        try:
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # Remove alpha channel
        except AttributeError:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_array
    
    def _create_attribute_chart(self, attributes: Dict, confidences: Dict) -> np.ndarray:
        """Create attribute prediction chart"""
        # Get top confident positive and negative attributes
        pos_attrs = [(k, confidences[k]) for k, v in attributes.items() if v == 1]
        neg_attrs = [(k, confidences[k]) for k, v in attributes.items() if v == 0]
        
        pos_attrs.sort(key=lambda x: x[1], reverse=True)
        neg_attrs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 of each
        top_attrs = pos_attrs[:5] + neg_attrs[:5]
        
        if not top_attrs:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [attr.replace('_', ' ') for attr, _ in top_attrs]
        values = [conf if attr in [a for a, _ in pos_attrs[:5]] else -conf 
                 for attr, conf in top_attrs]
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax.barh(names, values, color=colors)
        ax.set_xlabel('Confidence (Positive: Present, Negative: Absent)')
        ax.set_title('Top Confident Attribute Predictions')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Convert to image
        fig.canvas.draw()
        # Use buffer_rgba for newer matplotlib versions, fallback to tostring_rgb
        try:
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # Remove alpha channel
        except AttributeError:
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_array
    
    def _create_prototype_visualization(self, prototypes: List[Dict]) -> np.ndarray:
        """Create prototype comparison visualization"""
        if not prototypes:
            return np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Use prototype explainer's visualization method
        if 'prototypes' in self.explainers:
            return self.explainers['prototypes'].visualize_prototypes(prototypes[:6])
        
        return np.zeros((200, 200, 3), dtype=np.uint8)
    
    def _tensor_to_display_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable numpy image"""
        from .base import tensor_to_image
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        img = tensor_to_image(tensor)
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def build_prototype_database(self,
                               images: torch.Tensor,
                               identities: torch.Tensor,
                               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Build prototype database for prototype-based explanations"""
        if 'prototypes' not in self.explainers:
            raise ValueError("Prototype explainer not enabled")
        
        return self.explainers['prototypes'].build_prototype_database(
            images, identities, metadata
        )
    
    def learn_concept_vectors(self,
                            images: torch.Tensor,
                            attributes: torch.Tensor) -> Dict[str, Any]:
        """Learn concept activation vectors for TCAV analysis"""
        if 'tcav' not in self.explainers:
            raise ValueError("TCAV analyzer not enabled")
        
        return self.explainers['tcav'].learn_attribute_concepts(images, attributes)
    
    def save_explanation_database(self, save_dir: str):
        """Save all explanation databases and models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save prototype database
        if 'prototypes' in self.explainers:
            proto_path = os.path.join(save_dir, 'prototype_database.pkl')
            self.explainers['prototypes'].save_prototype_database(proto_path)
        
        # Save CAVs
        if 'tcav' in self.explainers:
            cav_path = os.path.join(save_dir, 'concept_vectors.pt')
            self.explainers['tcav'].save_cavs(cav_path)
        
        # Save configuration
        config = {
            'enabled_methods': self.enabled_methods,
            'explanation_style': self.explanation_style,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(save_dir, 'explainability_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Explanation databases saved to {save_dir}")
    
    def load_explanation_database(self, load_dir: str):
        """Load explanation databases"""
        import os
        
        # Load prototype database
        if 'prototypes' in self.explainers:
            proto_path = os.path.join(load_dir, 'prototype_database.pkl')
            if os.path.exists(proto_path):
                self.explainers['prototypes'].load_prototype_database(proto_path)
        
        # Load CAVs
        if 'tcav' in self.explainers:
            cav_path = os.path.join(load_dir, 'concept_vectors.pt')
            if os.path.exists(cav_path):
                self.explainers['tcav'].load_cavs(cav_path)
        
        print(f"Explanation databases loaded from {load_dir}")
    
    def generate_explanation_report(self, 
                                  results: List[ExplanationResult],
                                  save_path: Optional[str] = None) -> str:
        """Generate comprehensive explanation report"""
        if 'textual' not in self.explainers:
            raise ValueError("Textual explainer required for report generation")
        
        # Convert results to format expected by textual explainer
        explanations = []
        for result in results:
            exp_dict = {
                'similarity': result.match_score,
                'is_match': result.match_score > 0.5 if result.match_score else False,
                'confidence': result.explanation_confidence or 0.5,
                'components': getattr(result, 'detailed_results', {})
            }
            explanations.append(exp_dict)
        
        report = self.explainers['textual'].generate_summary_report(explanations)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def set_explanation_style(self, style: str):
        """Set explanation style for all explainers"""
        self.explanation_style = style
        
        if 'textual' in self.explainers:
            self.explainers['textual'].set_explanation_style(style)
    
    def _overlay_saliency(self, 
                         image: np.ndarray, 
                         saliency: np.ndarray, 
                         alpha: float = 0.4) -> np.ndarray:
        """Overlay saliency map on original image"""
        if len(image.shape) == 3 and image.shape[0] == 3:
            # Convert CHW to HWC
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = image * 255
        image = image.astype(np.uint8)
        
        # Ensure saliency is in correct format
        if len(saliency.shape) == 2:
            saliency = self._apply_colormap(saliency)
        
        # Resize saliency to match image size
        if saliency.shape[:2] != image.shape[:2]:
            saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))
        
        # Overlay
        return cv2.addWeighted(image, 1-alpha, saliency, alpha, 0)
    
    def _apply_colormap(self, saliency: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply colormap to saliency map"""
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency_uint8 = (saliency_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(saliency_uint8, colormap)