"""
Base classes for explainability components
==========================================

Defines abstract base classes and common utilities for all explanation methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import cv2
from PIL import Image


@dataclass
class ExplanationResult:
    """Container for explanation results"""
    
    # Basic prediction info
    embedding: np.ndarray
    identity_logits: Optional[np.ndarray] = None
    attribute_logits: Optional[np.ndarray] = None
    match_score: Optional[float] = None
    
    # Visual explanations
    saliency_map: Optional[np.ndarray] = None
    grad_cam: Optional[np.ndarray] = None
    integrated_gradients: Optional[np.ndarray] = None
    
    # Attribute explanations
    predicted_attributes: Optional[Dict[str, float]] = None
    attribute_confidences: Optional[Dict[str, float]] = None
    attribute_differences: Optional[Dict[str, float]] = None
    
    # Concept analysis
    concept_scores: Optional[Dict[str, float]] = None
    tcav_scores: Optional[Dict[str, float]] = None
    
    # Prototype explanations
    similar_prototypes: Optional[List[Dict]] = None
    dissimilar_prototypes: Optional[List[Dict]] = None
    prototype_similarities: Optional[List[float]] = None
    
    # Counterfactual analysis
    counterfactual_changes: Optional[Dict[str, float]] = None
    attribute_sensitivity: Optional[Dict[str, float]] = None
    
    # Textual explanation
    textual_explanation: Optional[str] = None
    explanation_confidence: Optional[float] = None


class BaseExplainer(ABC):
    """Abstract base class for all explainers"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize explainer
        
        Args:
            model: The face recognition model
            device: Device to run explanations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Register hooks for extracting features
        self.feature_maps = {}
        self.gradients = {}
        self._register_hooks()
    
    @abstractmethod
    def explain(self, 
                image: torch.Tensor, 
                target_class: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for the given image
        
        Args:
            image: Input image tensor
            target_class: Target class for explanation (if applicable)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing explanation results
        """
        pass
    
    def _register_hooks(self):
        """Register forward and backward hooks for feature extraction"""
        # This will be implemented by subclasses based on their needs
        pass
    
    def _normalize_saliency(self, saliency: np.ndarray) -> np.ndarray:
        """Normalize saliency map to [0, 1] range"""
        saliency = np.abs(saliency)
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency
    
    def _apply_colormap(self, 
                       saliency: np.ndarray, 
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply colormap to saliency map"""
        saliency_uint8 = (saliency * 255).astype(np.uint8)
        return cv2.applyColorMap(saliency_uint8, colormap)
    
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


class VerificationExplainer(BaseExplainer):
    """Specialized explainer for face verification tasks"""
    
    def explain_verification(self, 
                           image1: torch.Tensor, 
                           image2: torch.Tensor,
                           threshold: float = 0.5) -> ExplanationResult:
        """
        Explain face verification decision
        
        Args:
            image1: First face image
            image2: Second face image  
            threshold: Verification threshold
            
        Returns:
            Comprehensive explanation result
        """
        # Get embeddings and similarity
        with torch.no_grad():
            emb1 = self.model.get_embeddings(image1.unsqueeze(0))
            emb2 = self.model.get_embeddings(image2.unsqueeze(0))
            
        similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
        
        result = ExplanationResult(
            embedding=emb1.cpu().numpy().flatten(),
            match_score=similarity
        )
        
        return result


def get_attribute_names() -> List[str]:
    """Get list of CelebA attribute names"""
    return [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image array"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.shape[0] == 3:  # CHW format
        tensor = tensor.permute(1, 2, 0)  # Convert to HWC
    
    # Denormalize if needed (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = tensor.detach().cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image