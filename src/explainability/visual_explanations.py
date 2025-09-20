"""
Visual Explanation Methods
==========================

Implements Grad-CAM, Guided Grad-CAM, and Integrated Gradients for visual
explanations of face recognition and attribute prediction decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from .base import BaseExplainer, ExplanationResult, tensor_to_image


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM implementation for face recognition explanations
    
    Generates class activation maps showing which parts of the image
    the model focuses on for identity or attribute predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = 'backbone.layer4', device: str = 'cuda'):
        """
        Initialize Grad-CAM explainer
        
        Args:
            model: Face recognition model
            target_layer: Layer name to extract gradients from (default: ResNet layer4)
            device: Device to run on
        """
        self.target_layer = target_layer
        self.target_layer_gradients = None
        self.target_layer_activations = None
        super().__init__(model, device)
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks for target layer"""
        def forward_hook(module, input, output):
            self.target_layer_activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.target_layer_gradients = grad_output[0]
        
        # Find and register hooks on target layer
        target_module = self._get_target_layer()
        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get the target layer module"""
        try:
            # Navigate through model hierarchy
            parts = self.target_layer.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            print(f"Warning: Could not find target layer {self.target_layer}")
            return None
    
    def explain(self, 
                image: torch.Tensor, 
                target_type: str = 'identity',
                target_class: Optional[int] = None,
                attribute_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation
        
        Args:
            image: Input image tensor [1, 3, H, W]
            target_type: 'identity' or 'attribute'
            target_class: Target identity class (for identity explanations)
            attribute_idx: Target attribute index (for attribute explanations)
            
        Returns:
            Dictionary containing Grad-CAM results
        """
        self.model.eval()
        image = image.to(self.device).requires_grad_(True)
        
        # Forward pass
        if hasattr(self.model, 'forward_with_cam'):
            outputs = self.model.forward_with_cam(image)
        else:
            outputs = self.model(image)
        
        # Select target for explanation
        if target_type == 'identity':
            if 'identity_logits' in outputs:
                logits = outputs['identity_logits']
            else:
                # Fallback to embeddings
                logits = outputs['embeddings']
                
            if target_class is None:
                target_class = logits.argmax(dim=1)
            target_score = logits[0, target_class] if logits.dim() > 1 else logits.mean()
            
        elif target_type == 'attribute':
            if 'attribute_logits' in outputs:
                logits = outputs['attribute_logits']
                if attribute_idx is None:
                    attribute_idx = 0  # Default to first attribute
                target_score = logits[0, attribute_idx]
            else:
                raise ValueError("Model does not output attribute logits")
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Generate Grad-CAM
        grad_cam = self._generate_gradcam()
        
        # Convert to numpy and overlay
        image_np = tensor_to_image(image)
        grad_cam_overlay = self._overlay_saliency(image_np, grad_cam)
        
        return {
            'grad_cam': grad_cam,
            'grad_cam_overlay': grad_cam_overlay,
            'target_score': target_score.item(),
            'target_type': target_type,
            'target_class': target_class,
            'attribute_idx': attribute_idx
        }
    
    def _generate_gradcam(self) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        if self.target_layer_gradients is None or self.target_layer_activations is None:
            print("Warning: No gradients or activations found. Using dummy heatmap.")
            return np.ones((224, 224)) * 0.5
        
        # Get gradients and activations
        gradients = self.target_layer_gradients
        activations = self.target_layer_activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        grad_cam = F.relu(grad_cam)
        
        # Normalize
        grad_cam = grad_cam.squeeze().detach().cpu().numpy()
        grad_cam = self._normalize_saliency(grad_cam)
        
        # Resize to input image size
        grad_cam = cv2.resize(grad_cam, (224, 224))
        
        return grad_cam
    
    def explain_verification(self, 
                           image1: torch.Tensor, 
                           image2: torch.Tensor,
                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanations for face verification
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Verification threshold
            
        Returns:
            Verification explanation with Grad-CAMs for both images
        """
        # Get embeddings and similarity
        with torch.no_grad():
            # Ensure single images (squeeze if batched)
            img1 = image1.squeeze(0) if image1.dim() == 4 and image1.size(0) == 1 else image1
            img2 = image2.squeeze(0) if image2.dim() == 4 and image2.size(0) == 1 else image2
            
            emb1 = self.model.get_embeddings(img1.unsqueeze(0))
            emb2 = self.model.get_embeddings(img2.unsqueeze(0))
            
        similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
        is_match = similarity > threshold
        
        # Generate Grad-CAM for embedding similarity
        # This requires a custom forward pass that optimizes similarity
        explanation1 = self._explain_for_similarity(img1, emb2)
        explanation2 = self._explain_for_similarity(img2, emb1)
        
        return {
            'similarity': similarity,
            'is_match': is_match,
            'threshold': threshold,
            'image1_gradcam': explanation1,
            'image2_gradcam': explanation2
        }
    
    def _explain_for_similarity(self, image: torch.Tensor, target_embedding: torch.Tensor) -> Dict[str, Any]:
        """Generate explanation for embedding similarity"""
        image = image.to(self.device).requires_grad_(True)
        
        # Forward pass to get embedding
        embedding = self.model.get_embeddings(image.unsqueeze(0))
        
        # Compute similarity with target
        similarity = torch.cosine_similarity(embedding, target_embedding, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        similarity.backward(retain_graph=True)
        
        # Generate Grad-CAM
        grad_cam = self._generate_gradcam()
        
        # Convert to overlay
        image_np = tensor_to_image(image)
        grad_cam_overlay = self._overlay_saliency(image_np, grad_cam)
        
        return {
            'grad_cam': grad_cam,
            'grad_cam_overlay': grad_cam_overlay,
            'similarity': similarity.item()
        }


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients implementation for axiomatic attributions
    
    Provides pixel-level importance scores with theoretical guarantees
    about completeness and sensitivity.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Integrated Gradients explainer
        
        Args:
            model: Face recognition model
            device: Device to run on
        """
        super().__init__(model, device)
    
    def explain(self, 
                image: torch.Tensor,
                target_type: str = 'identity', 
                target_class: Optional[int] = None,
                attribute_idx: Optional[int] = None,
                steps: int = 50,
                baseline: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanation
        
        Args:
            image: Input image tensor [1, 3, H, W]
            target_type: 'identity' or 'attribute'
            target_class: Target identity class
            attribute_idx: Target attribute index
            steps: Number of integration steps
            baseline: Baseline image (default: zeros)
            
        Returns:
            Dictionary containing Integrated Gradients results
        """
        self.model.eval()
        image = image.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        else:
            baseline = baseline.to(self.device)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)
        
        # Compute gradients along path
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(interpolated)
            
            # Select target
            if target_type == 'identity':
                logits = outputs.get('identity_logits', outputs.get('embeddings'))
                if target_class is None:
                    target_class = logits.argmax(dim=1)
                target_score = logits[0, target_class] if logits.dim() > 1 else logits.mean()
            elif target_type == 'attribute':
                logits = outputs['attribute_logits']
                if attribute_idx is None:
                    attribute_idx = 0
                target_score = logits[0, attribute_idx]
            
            # Compute gradients
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            gradients.append(interpolated.grad.cpu().clone())
        
        # Integrate gradients using trapezoidal rule
        integrated_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = integrated_gradients * (image.cpu() - baseline.cpu())
        
        # Convert to attribution map
        attribution = integrated_gradients.squeeze().abs().sum(dim=0).cpu().numpy()
        attribution = self._normalize_saliency(attribution)
        
        # Create overlay
        image_np = tensor_to_image(image)
        attribution_overlay = self._overlay_saliency(image_np, attribution)
        
        return {
            'integrated_gradients': attribution,
            'attribution_overlay': attribution_overlay,
            'target_type': target_type,
            'target_class': target_class,
            'attribute_idx': attribute_idx,
            'steps': steps
        }


class GuidedGradCAMExplainer(GradCAMExplainer):
    """
    Guided Grad-CAM combines Grad-CAM with Guided Backpropagation
    for more precise localization
    """
    
    def __init__(self, model: nn.Module, target_layer: str = 'backbone.layer4', device: str = 'cuda'):
        super().__init__(model, target_layer, device)
        self._register_guided_hooks()
    
    def _register_guided_hooks(self):
        """Register hooks for guided backpropagation"""
        def guided_relu_hook(module, grad_input, grad_output):
            # Only keep positive gradients (guided backpropagation)
            return (torch.clamp(grad_input[0], min=0.0),)
        
        # Apply to all ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(guided_relu_hook)
    
    def explain(self, 
                image: torch.Tensor, 
                target_type: str = 'identity',
                target_class: Optional[int] = None,
                attribute_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate Guided Grad-CAM explanation
        
        Returns both Grad-CAM and Guided Grad-CAM results
        """
        # Get regular Grad-CAM
        gradcam_result = super().explain(image, target_type, target_class, attribute_idx)
        
        # Get guided backpropagation
        guided_result = self._guided_backprop(image, target_type, target_class, attribute_idx)
        
        # Combine Grad-CAM with guided backpropagation
        grad_cam = gradcam_result['grad_cam']
        guided_grads = guided_result['guided_gradients']
        
        # Element-wise multiplication
        guided_gradcam = grad_cam * guided_grads
        guided_gradcam = self._normalize_saliency(guided_gradcam)
        
        # Create overlay
        image_np = tensor_to_image(image)
        guided_overlay = self._overlay_saliency(image_np, guided_gradcam)
        
        return {
            **gradcam_result,
            'guided_gradients': guided_grads,
            'guided_gradcam': guided_gradcam,
            'guided_overlay': guided_overlay
        }
    
    def _guided_backprop(self, 
                        image: torch.Tensor,
                        target_type: str,
                        target_class: Optional[int],
                        attribute_idx: Optional[int]) -> Dict[str, Any]:
        """Perform guided backpropagation"""
        image = image.to(self.device).requires_grad_(True)
        
        # Forward pass
        outputs = self.model(image)
        
        # Select target
        if target_type == 'identity':
            logits = outputs.get('identity_logits', outputs.get('embeddings'))
            if target_class is None:
                target_class = logits.argmax(dim=1)
            target_score = logits[0, target_class] if logits.dim() > 1 else logits.mean()
        elif target_type == 'attribute':
            logits = outputs['attribute_logits']
            if attribute_idx is None:
                attribute_idx = 0
            target_score = logits[0, attribute_idx]
        
        # Backward pass with guided hooks active
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get guided gradients
        guided_gradients = image.grad.squeeze().abs().sum(dim=0).cpu().numpy()
        guided_gradients = self._normalize_saliency(guided_gradients)
        
        return {
            'guided_gradients': guided_gradients,
            'target_score': target_score.item()
        }