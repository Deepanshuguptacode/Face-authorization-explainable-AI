"""
Loss Functions for Multi-task Face Recognition
==============================================

This module implements various loss functions:
- CrossEntropy loss for identity (used with ArcFace)
- Binary Cross Entropy for binary attributes
- Focal Loss for handling class imbalance
- Multi-task loss combination with adaptive weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for handling class imbalance"""
    
    def __init__(self, pos_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Args:
            pos_weights: Positive class weights [num_attributes]
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [batch_size, num_attributes]
            targets: Ground truth labels [batch_size, num_attributes] (values in {-1, 1})
        
        Returns:
            Weighted BCE loss
        """
        # Convert {-1, 1} to {0, 1}
        targets_binary = (targets + 1) / 2
        
        if self.pos_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets_binary, 
                pos_weight=self.pos_weights.to(inputs.device),
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets_binary, 
                reduction=self.reduction
            )
        
        return loss


class AdaptiveTaskWeighting(nn.Module):
    """
    Adaptive task weighting for multi-task learning
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self, num_tasks: int, init_log_vars: Optional[List[float]] = None):
        """
        Args:
            num_tasks: Number of tasks
            init_log_vars: Initial log variance values for each task
        """
        super(AdaptiveTaskWeighting, self).__init__()
        
        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks
        
        # Learnable log variance parameters
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: List of task-specific losses
        
        Returns:
            Tuple of (weighted_loss, weights_dict)
        """
        weighted_losses = []
        weights = {}
        
        for i, loss in enumerate(losses):
            # Compute weight: 1 / (2 * variance) = 1 / (2 * exp(log_var))
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            
            # Store weights for logging
            weights[f'task_{i}_weight'] = precision.item()
            weights[f'task_{i}_log_var'] = self.log_vars[i].item()
        
        total_loss = sum(weighted_losses)
        return total_loss, weights


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining identity and attribute losses
    """
    
    def __init__(self,
                 num_attributes: int,
                 attribute_weights: Optional[torch.Tensor] = None,
                 use_focal_identity: bool = False,
                 use_adaptive_weighting: bool = True,
                 identity_weight: float = 1.0,
                 attribute_weight: float = 1.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Args:
            num_attributes: Number of facial attributes
            attribute_weights: Per-attribute positive class weights
            use_focal_identity: Use focal loss for identity classification
            use_adaptive_weighting: Use adaptive task weighting
            identity_weight: Fixed weight for identity loss (if not adaptive)
            attribute_weight: Fixed weight for attribute loss (if not adaptive)
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super(MultiTaskLoss, self).__init__()
        
        self.num_attributes = num_attributes
        self.use_adaptive_weighting = use_adaptive_weighting
        self.identity_weight = identity_weight
        self.attribute_weight = attribute_weight
        
        # Identity loss
        if use_focal_identity:
            self.identity_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.identity_loss = nn.CrossEntropyLoss()
        
        # Attribute loss
        self.attribute_loss = WeightedBCELoss(pos_weights=attribute_weights)
        
        # Adaptive weighting
        if use_adaptive_weighting:
            self.adaptive_weighting = AdaptiveTaskWeighting(num_tasks=2)
        else:
            self.adaptive_weighting = None
    
    def forward(self,
                identity_logits: torch.Tensor,
                attribute_logits: torch.Tensor,
                identity_labels: torch.Tensor,
                attribute_labels: torch.Tensor,
                train_identity: bool = True,
                train_attributes: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            identity_logits: Identity predictions [batch_size, num_classes]
            attribute_logits: Attribute predictions [batch_size, num_attributes]
            identity_labels: Identity ground truth [batch_size]
            attribute_labels: Attribute ground truth [batch_size, num_attributes]
            train_identity: Whether to compute identity loss
            train_attributes: Whether to compute attribute loss
        
        Returns:
            Dictionary containing losses and metrics
        """
        losses = {}
        task_losses = []
        
        # Identity loss
        if train_identity:
            identity_loss = self.identity_loss(identity_logits, identity_labels)
            losses['identity_loss'] = identity_loss
            task_losses.append(identity_loss)
        
        # Attribute loss  
        if train_attributes:
            attribute_loss = self.attribute_loss(attribute_logits, attribute_labels)
            losses['attribute_loss'] = attribute_loss
            task_losses.append(attribute_loss)
        
        # Combine losses
        if len(task_losses) > 1:
            if self.use_adaptive_weighting:
                total_loss, weights = self.adaptive_weighting(task_losses)
                losses.update(weights)
            else:
                total_loss = (
                    self.identity_weight * losses.get('identity_loss', 0) +
                    self.attribute_weight * losses.get('attribute_loss', 0)
                )
        elif len(task_losses) == 1:
            total_loss = task_losses[0]
        else:
            total_loss = torch.tensor(0.0, device=identity_logits.device)
        
        losses['total_loss'] = total_loss
        
        return losses


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for negative pairs
            reduction: Reduction method
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, 
                embeddings1: torch.Tensor, 
                embeddings2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings1, embeddings2: Face embeddings [batch_size, embedding_dim]
            labels: Binary labels (1 for same person, 0 for different)
        
        Returns:
            Contrastive loss
        """
        # Euclidean distance
        distance = F.pairwise_distance(embeddings1, embeddings2, p=2)
        
        # Contrastive loss
        positive_loss = labels * torch.pow(distance, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = positive_loss + negative_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """Triplet loss for face recognition"""
    
    def __init__(self, margin: float = 0.3, p: int = 2, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for triplet loss
            p: Norm degree for distance calculation
            reduction: Reduction method
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor, positive, negative: Face embeddings [batch_size, embedding_dim]
        
        Returns:
            Triplet loss
        """
        # Calculate distances
        pos_distance = F.pairwise_distance(anchor, positive, p=self.p)
        neg_distance = F.pairwise_distance(anchor, negative, p=self.p)
        
        # Triplet loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_attribute_weights(attribute_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute positive class weights for attributes based on class frequency
    
    Args:
        attribute_labels: Attribute labels [num_samples, num_attributes] with values {-1, 1}
    
    Returns:
        Positive class weights [num_attributes]
    """
    # Convert to binary {0, 1}
    binary_labels = (attribute_labels + 1) / 2
    
    # Compute positive ratios
    pos_ratios = binary_labels.mean(dim=0)
    
    # Compute weights (inverse frequency)
    weights = 1.0 / (pos_ratios + 1e-8)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    return weights


def create_loss_function(config: Dict, attribute_labels: Optional[torch.Tensor] = None) -> MultiTaskLoss:
    """
    Factory function to create loss function from config
    
    Args:
        config: Loss configuration dictionary
        attribute_labels: Attribute labels for computing weights
    
    Returns:
        Multi-task loss function
    """
    # Compute attribute weights if provided
    attribute_weights = None
    if attribute_labels is not None:
        attribute_weights = compute_attribute_weights(attribute_labels)
    
    return MultiTaskLoss(
        num_attributes=config.get('num_attributes', 40),
        attribute_weights=attribute_weights,
        use_focal_identity=config.get('use_focal_identity', False),
        use_adaptive_weighting=config.get('use_adaptive_weighting', True),
        identity_weight=config.get('identity_weight', 1.0),
        attribute_weight=config.get('attribute_weight', 1.0),
        focal_alpha=config.get('focal_alpha', 1.0),
        focal_gamma=config.get('focal_gamma', 2.0)
    )


# Loss configuration presets
LOSS_CONFIGS = {
    'baseline': {
        'num_attributes': 40,
        'use_focal_identity': False,
        'use_adaptive_weighting': True,
        'identity_weight': 1.0,
        'attribute_weight': 1.0
    },
    'focal': {
        'num_attributes': 40,
        'use_focal_identity': True,
        'use_adaptive_weighting': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0
    },
    'fixed_weights': {
        'num_attributes': 40,
        'use_focal_identity': False,
        'use_adaptive_weighting': False,
        'identity_weight': 1.0,
        'attribute_weight': 0.5
    }
}


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    num_classes = 1000
    num_attributes = 40
    embedding_dim = 512
    
    # Create dummy data
    identity_logits = torch.randn(batch_size, num_classes)
    attribute_logits = torch.randn(batch_size, num_attributes)
    identity_labels = torch.randint(0, num_classes, (batch_size,))
    attribute_labels = torch.randint(0, 2, (batch_size, num_attributes)) * 2 - 1  # {-1, 1}
    
    # Create loss function
    config = LOSS_CONFIGS['baseline']
    loss_fn = create_loss_function(config, attribute_labels)
    
    # Test forward pass
    losses = loss_fn(
        identity_logits=identity_logits,
        attribute_logits=attribute_logits,
        identity_labels=identity_labels,
        attribute_labels=attribute_labels
    )
    
    print("Loss Function Test:")
    for key, value in losses.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # Test attribute weights computation
    weights = compute_attribute_weights(attribute_labels)
    print(f"\\nAttribute weights shape: {weights.shape}")
    print(f"Attribute weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    print("✓ Loss function test passed!")