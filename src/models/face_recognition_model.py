"""
Face Recognition Model Architecture
===================================

This module implements the multi-task face recognition model with:
- ResNet-50 backbone
- Identity head with ArcFace loss
- Attribute heads for explainability
- Optional Siamese verification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import math
from typing import Dict, List, Optional, Tuple


class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 scale: float = 64.0,
                 margin: float = 0.5,
                 easy_margin: bool = False):
        """
        Args:
            in_features: Size of input features (embedding dimension)
            out_features: Number of classes (identities)
            scale: Scaling factor for logits
            margin: Angular margin penalty
            easy_margin: Use easy margin boundary
        """
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin components
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, input: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input: Input embeddings [batch_size, in_features]
            label: Ground truth labels [batch_size] (only needed during training)
        
        Returns:
            Scaled logits [batch_size, out_features]
        """
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:  # Inference mode
            return cosine * self.scale
            
        # Training mode - apply angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding for labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin only to positive samples
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class AttributeHead(nn.Module):
    """Multi-label attribute prediction head"""
    
    def __init__(self, 
                 in_features: int,
                 num_attributes: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            in_features: Size of input features
            num_attributes: Number of attributes to predict
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(AttributeHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_attributes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, in_features]
        
        Returns:
            Attribute logits [batch_size, num_attributes]
        """
        return self.classifier(x)


class SiameseHead(nn.Module):
    """Optional Siamese head for pairwise verification"""
    
    def __init__(self, in_features: int, hidden_dim: int = 512):
        super(SiameseHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1, emb2: Face embeddings [batch_size, embedding_dim]
        
        Returns:
            Similarity scores [batch_size, 1]
        """
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        return self.projection(combined)


class FaceRecognitionModel(nn.Module):
    """
    Multi-task Face Recognition Model
    
    Architecture:
    - ResNet-50 backbone (shared features)
    - Identity head with ArcFace loss
    - Attribute heads for explainability
    - Optional Siamese verification head
    """
    
    def __init__(self,
                 num_classes: int,
                 num_attributes: int,
                 embedding_dim: int = 512,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 use_arcface: bool = True,
                 use_siamese: bool = False,
                 arcface_scale: float = 64.0,
                 arcface_margin: float = 0.5,
                 attribute_hidden_dim: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            num_classes: Number of identity classes
            num_attributes: Number of facial attributes
            embedding_dim: Face embedding dimension
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            use_arcface: Use ArcFace head for identity
            use_siamese: Add Siamese verification head
            arcface_scale: ArcFace scaling factor
            arcface_margin: ArcFace margin
            attribute_hidden_dim: Hidden dimension for attribute heads
            dropout: Dropout probability
        """
        super(FaceRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.embedding_dim = embedding_dim
        self.use_arcface = use_arcface
        self.use_siamese = use_siamese
        
        # Backbone network
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        else:
            # Use timm for other architectures
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            backbone_dim = self.backbone.num_features
        
        # Feature projection to embedding space
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Identity head
        if self.use_arcface:
            self.identity_head = ArcFaceHead(
                in_features=embedding_dim,
                out_features=num_classes,
                scale=arcface_scale,
                margin=arcface_margin
            )
        else:
            self.identity_head = nn.Linear(embedding_dim, num_classes)
        
        # Attribute heads
        self.attribute_head = AttributeHead(
            in_features=embedding_dim,
            num_attributes=num_attributes,
            hidden_dim=attribute_hidden_dim,
            dropout=dropout
        )
        
        # Optional Siamese head
        if self.use_siamese:
            self.siamese_head = SiameseHead(
                in_features=embedding_dim,
                hidden_dim=attribute_hidden_dim
            )
        
    def forward(self, 
                x: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
            labels: Identity labels [batch_size] (for ArcFace training)
            return_embeddings: Whether to return face embeddings
        
        Returns:
            Dictionary containing:
            - identity_logits: Identity classification logits
            - attribute_logits: Attribute prediction logits
            - embeddings: Face embeddings (if return_embeddings=True)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.feature_projection(features)
        
        # Identity prediction
        if self.use_arcface and self.training and labels is not None:
            identity_logits = self.identity_head(embeddings, labels)
        else:
            identity_logits = self.identity_head(embeddings)
        
        # Attribute prediction
        attribute_logits = self.attribute_head(embeddings)
        
        # Prepare output
        output = {
            'identity_logits': identity_logits,
            'attribute_logits': attribute_logits
        }
        
        if return_embeddings:
            output['embeddings'] = embeddings
        
        return output
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings for inference"""
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.feature_projection(features)
            return F.normalize(embeddings, p=2, dim=1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for extract_embeddings for explainability compatibility"""
        return self.extract_embeddings(x)
    
    def verify_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Verify if two faces belong to the same person"""
        emb1 = self.extract_embeddings(x1)
        emb2 = self.extract_embeddings(x2)
        
        if self.use_siamese:
            return self.siamese_head(emb1, emb2)
        else:
            # Use cosine similarity
            similarity = F.cosine_similarity(emb1, emb2, dim=1)
            return (similarity + 1) / 2  # Normalize to [0, 1]


def create_model(config: Dict) -> FaceRecognitionModel:
    """
    Factory function to create model from config
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    return FaceRecognitionModel(
        num_classes=config.get('num_classes', 1000),
        num_attributes=config.get('num_attributes', 40),
        embedding_dim=config.get('embedding_dim', 512),
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        use_arcface=config.get('use_arcface', True),
        use_siamese=config.get('use_siamese', False),
        arcface_scale=config.get('arcface_scale', 64.0),
        arcface_margin=config.get('arcface_margin', 0.5),
        attribute_hidden_dim=config.get('attribute_hidden_dim', 512),
        dropout=config.get('dropout', 0.1)
    )


# Model configuration presets
MODEL_CONFIGS = {
    'baseline': {
        'num_classes': 10177,  # Unique identities in CelebA
        'num_attributes': 40,
        'embedding_dim': 512,
        'backbone': 'resnet50',
        'pretrained': True,
        'use_arcface': True,
        'use_siamese': False,
        'arcface_scale': 64.0,
        'arcface_margin': 0.5,
        'attribute_hidden_dim': 512,
        'dropout': 0.1
    },
    'identity_only': {
        'num_classes': 10177,
        'num_attributes': 40,
        'embedding_dim': 512,
        'backbone': 'resnet50',
        'pretrained': True,
        'use_arcface': True,
        'use_siamese': False,
        'train_attributes': False  # Will be handled in training script
    },
    'attributes_only': {
        'num_classes': 10177,
        'num_attributes': 40,
        'embedding_dim': 512,
        'backbone': 'resnet50',
        'pretrained': True,
        'use_arcface': False,
        'use_siamese': False,
        'train_identity': False  # Will be handled in training script
    }
}


if __name__ == "__main__":
    # Test model creation
    config = MODEL_CONFIGS['baseline']
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, config['num_classes'], (batch_size,))
    
    model.train()
    output = model(x, labels, return_embeddings=True)
    
    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Identity logits shape: {output['identity_logits'].shape}")
    print(f"Attribute logits shape: {output['attribute_logits'].shape}")
    print(f"Embeddings shape: {output['embeddings'].shape}")
    
    # Test inference
    model.eval()
    embeddings = model.extract_embeddings(x)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Test verification
    x1, x2 = x[:2], x[2:4]
    similarity = model.verify_pair(x1, x2)
    print(f"Verification similarity shape: {similarity.shape}")
    
    print("✓ Model architecture test passed!")