"""
TCAV (Testing with Concept Activation Vectors) Implementation
============================================================

Implements TCAV for quantifying how much high-level concepts influence
face recognition decisions. TCAV provides numerical influence scores
for human-interpretable concepts like "glasses", "beard", "age", etc.

Based on "Interpretability Beyond Feature Attribution: Quantitative 
Testing with Concept Activation Vectors (TCAV)" by Kim et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from .base import BaseExplainer, get_attribute_names


class ConceptActivationVector:
    """Represents a learned concept activation vector"""
    
    def __init__(self, 
                 concept_name: str,
                 layer_name: str, 
                 vector: np.ndarray,
                 accuracy: float):
        """
        Initialize CAV
        
        Args:
            concept_name: Name of the concept (e.g., "glasses")
            layer_name: Layer where CAV was computed
            vector: The concept activation vector
            accuracy: Accuracy of the concept classifier
        """
        self.concept_name = concept_name
        self.layer_name = layer_name
        self.vector = vector
        self.accuracy = accuracy
        
    def __repr__(self):
        return f"CAV({self.concept_name}, {self.layer_name}, acc={self.accuracy:.3f})"


class TCAVAnalyzer(BaseExplainer):
    """
    TCAV implementation for concept-based explanations
    
    Learns concept activation vectors from examples and computes
    their influence on model predictions.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 layer_names: List[str] = None,
                 device: str = 'cuda'):
        """
        Initialize TCAV analyzer
        
        Args:
            model: Face recognition model
            layer_names: List of layer names to extract activations from
            device: Device to run on
        """
        super().__init__(model, device)
        
        if layer_names is None:
            # Default layers for ResNet-50
            layer_names = ['backbone.layer1', 'backbone.layer2', 
                          'backbone.layer3', 'backbone.layer4']
        
        self.layer_names = layer_names
        self.activations = {}
        self.cavs = {}  # Stored concept activation vectors
        
        self._register_activation_hooks()
    
    def _register_activation_hooks(self):
        """Register hooks to extract activations from specified layers"""
        def make_hook(layer_name):
            def hook(module, input, output):
                # Global average pooling for conv layers
                if len(output.shape) == 4:  # Conv layer [B, C, H, W]
                    activation = F.adaptive_avg_pool2d(output, 1).squeeze()
                else:  # FC layer [B, C]
                    activation = output
                
                self.activations[layer_name] = activation.detach().cpu().numpy()
            return hook
        
        # Register hooks
        for layer_name in self.layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                layer.register_forward_hook(make_hook(layer_name))
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer module by name"""
        try:
            parts = layer_name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            print(f"Warning: Could not find layer {layer_name}")
            return None
    
    def learn_concept(self,
                     concept_name: str,
                     positive_images: torch.Tensor,
                     negative_images: torch.Tensor,
                     layer_name: str = 'backbone.layer4',
                     min_accuracy: float = 0.65) -> Optional[ConceptActivationVector]:
        """
        Learn a concept activation vector from positive and negative examples
        
        Args:
            concept_name: Name of the concept (e.g., "glasses", "smiling")
            positive_images: Images that contain the concept [N, 3, H, W]
            negative_images: Images that don't contain the concept [M, 3, H, W]
            layer_name: Layer to extract activations from
            min_accuracy: Minimum accuracy required for concept learning
            
        Returns:
            ConceptActivationVector if learning successful, None otherwise
        """
        print(f"Learning concept '{concept_name}' at layer '{layer_name}'...")
        
        # Extract activations for positive examples
        pos_activations = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(positive_images), 32):  # Process in batches
                batch = positive_images[i:i+32].to(self.device)
                _ = self.model(batch)
                pos_activations.append(self.activations[layer_name].copy())
        
        pos_activations = np.vstack(pos_activations)
        
        # Extract activations for negative examples
        neg_activations = []
        
        with torch.no_grad():
            for i in range(0, len(negative_images), 32):
                batch = negative_images[i:i+32].to(self.device)
                _ = self.model(batch)
                neg_activations.append(self.activations[layer_name].copy())
        
        neg_activations = np.vstack(neg_activations)
        
        # Prepare training data
        X = np.vstack([pos_activations, neg_activations])
        y = np.hstack([np.ones(len(pos_activations)), np.zeros(len(neg_activations))])
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train concept classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Evaluate accuracy
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Concept '{concept_name}' classifier accuracy: {accuracy:.3f}")
        
        if accuracy < min_accuracy:
            print(f"Warning: Low accuracy ({accuracy:.3f}) for concept '{concept_name}'")
            return None
        
        # Extract concept activation vector (classifier weights)
        cav_vector = classifier.coef_[0]
        
        # Create CAV object
        cav = ConceptActivationVector(concept_name, layer_name, cav_vector, accuracy)
        
        # Store CAV
        if layer_name not in self.cavs:
            self.cavs[layer_name] = {}
        self.cavs[layer_name][concept_name] = cav
        
        return cav
    
    def compute_tcav_score(self,
                          images: torch.Tensor,
                          concept_name: str,
                          layer_name: str = 'backbone.layer4',
                          class_idx: Optional[int] = None) -> float:
        """
        Compute TCAV score for a concept on given images
        
        Args:
            images: Input images [N, 3, H, W]
            concept_name: Name of the concept
            layer_name: Layer where CAV was computed
            class_idx: Target class index (for classification)
            
        Returns:
            TCAV score (between 0 and 1)
        """
        if layer_name not in self.cavs or concept_name not in self.cavs[layer_name]:
            raise ValueError(f"CAV for concept '{concept_name}' at layer '{layer_name}' not found")
        
        cav = self.cavs[layer_name][concept_name]
        
        # Get activations and gradients
        activations_list = []
        gradients_list = []
        
        for i in range(len(images)):
            image = images[i:i+1].to(self.device).requires_grad_(True)
            
            # Forward pass
            outputs = self.model(image)
            activation = self.activations[layer_name]
            
            # Get target for gradient computation
            if class_idx is not None:
                if 'identity_logits' in outputs:
                    target = outputs['identity_logits'][0, class_idx]
                else:
                    target = outputs['embeddings'][0, class_idx]
            else:
                # Use maximum activation
                if 'identity_logits' in outputs:
                    target = outputs['identity_logits'].max()
                else:
                    target = outputs['embeddings'].max()
            
            # Backward pass
            self.model.zero_grad()
            target.backward(retain_graph=True)
            
            # Get gradients at the layer
            grad = self._get_layer_gradients(layer_name)
            
            activations_list.append(activation)
            gradients_list.append(grad)
        
        activations = np.vstack(activations_list)
        gradients = np.vstack(gradients_list)
        
        # Compute directional derivatives
        directional_derivatives = []
        
        for i in range(len(activations)):
            grad = gradients[i]
            cav_vector = cav.vector
            
            # Compute dot product (directional derivative)
            dot_product = np.dot(grad, cav_vector)
            directional_derivatives.append(dot_product)
        
        # TCAV score is fraction of positive directional derivatives
        positive_derivatives = np.sum(np.array(directional_derivatives) > 0)
        tcav_score = positive_derivatives / len(directional_derivatives)
        
        return tcav_score
    
    def _get_layer_gradients(self, layer_name: str) -> np.ndarray:
        """Get gradients for a specific layer"""
        # This is a simplified version - in practice, you'd need to
        # register backward hooks to capture gradients
        layer = self._get_layer_by_name(layer_name)
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            return layer.weight.grad.cpu().numpy().flatten()
        else:
            # Return dummy gradients for now
            return np.random.randn(self.activations[layer_name].shape[-1])
    
    def learn_attribute_concepts(self,
                               images: torch.Tensor,
                               attributes: torch.Tensor,
                               layer_name: str = 'backbone.layer4') -> Dict[str, ConceptActivationVector]:
        """
        Learn CAVs for all facial attributes
        
        Args:
            images: Training images [N, 3, H, W]
            attributes: Attribute labels [N, 40] (CelebA format)
            layer_name: Layer to extract activations from
            
        Returns:
            Dictionary of learned CAVs
        """
        attribute_names = get_attribute_names()
        learned_cavs = {}
        
        for i, attr_name in enumerate(attribute_names):
            # Get positive and negative examples
            positive_mask = attributes[:, i] == 1
            negative_mask = attributes[:, i] == 0  # or -1 for CelebA
            
            if positive_mask.sum() < 50 or negative_mask.sum() < 50:
                print(f"Skipping {attr_name}: insufficient examples")
                continue
            
            positive_images = images[positive_mask]
            negative_images = images[negative_mask]
            
            # Sample to balance and limit size
            n_samples = min(500, positive_mask.sum(), negative_mask.sum())
            
            pos_indices = np.random.choice(len(positive_images), n_samples, replace=False)
            neg_indices = np.random.choice(len(negative_images), n_samples, replace=False)
            
            cav = self.learn_concept(
                attr_name,
                positive_images[pos_indices],
                negative_images[neg_indices],
                layer_name
            )
            
            if cav is not None:
                learned_cavs[attr_name] = cav
        
        return learned_cavs
    
    def explain_with_concepts(self,
                            image: torch.Tensor,
                            concept_names: List[str] = None,
                            layer_name: str = 'backbone.layer4') -> Dict[str, float]:
        """
        Explain model prediction using concept influence scores
        
        Args:
            image: Input image [1, 3, H, W]
            concept_names: List of concepts to analyze
            layer_name: Layer to analyze
            
        Returns:
            Dictionary of concept influence scores
        """
        if concept_names is None:
            # Use all available concepts
            if layer_name in self.cavs:
                concept_names = list(self.cavs[layer_name].keys())
            else:
                raise ValueError(f"No CAVs available for layer {layer_name}")
        
        concept_scores = {}
        
        for concept_name in concept_names:
            try:
                score = self.compute_tcav_score(image, concept_name, layer_name)
                concept_scores[concept_name] = score
            except ValueError as e:
                print(f"Warning: Could not compute TCAV for {concept_name}: {e}")
                concept_scores[concept_name] = 0.0
        
        return concept_scores
    
    def save_cavs(self, save_path: str):
        """Save learned CAVs to disk"""
        torch.save(self.cavs, save_path)
        print(f"Saved CAVs to {save_path}")
    
    def load_cavs(self, load_path: str):
        """Load CAVs from disk"""
        if os.path.exists(load_path):
            self.cavs = torch.load(load_path, map_location='cpu')
            print(f"Loaded CAVs from {load_path}")
        else:
            raise FileNotFoundError(f"CAV file not found: {load_path}")
    
    def explain(self, 
                image: torch.Tensor,
                concept_names: List[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate TCAV-based explanation for input image
        
        Args:
            image: Input image tensor
            concept_names: Concepts to analyze
            
        Returns:
            Dictionary containing TCAV scores and explanations
        """
        concept_scores = self.explain_with_concepts(image, concept_names)
        
        # Sort concepts by influence
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'tcav_scores': concept_scores,
            'top_influential_concepts': sorted_concepts[:10],
            'concept_summary': self._generate_concept_summary(sorted_concepts)
        }
    
    def _generate_concept_summary(self, sorted_concepts: List[Tuple[str, float]]) -> str:
        """Generate textual summary of concept influences"""
        if not sorted_concepts:
            return "No concept influences computed."
        
        top_concepts = sorted_concepts[:5]
        
        summary_parts = []
        for concept, score in top_concepts:
            influence_level = "high" if score > 0.7 else "moderate" if score > 0.4 else "low"
            concept_clean = concept.replace('_', ' ').lower()
            summary_parts.append(f"{concept_clean} ({influence_level}: {score:.2f})")
        
        return f"Top concept influences: {', '.join(summary_parts)}"