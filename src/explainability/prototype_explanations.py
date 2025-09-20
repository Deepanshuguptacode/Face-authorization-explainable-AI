"""
Prototype-based Explanations
============================

Provides example-based explanations by finding nearest prototype embeddings
and similar/dissimilar images. Helps users understand model decisions through
concrete examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import cv2
from .base import BaseExplainer, ExplanationResult, tensor_to_image


class PrototypeExplainer(BaseExplainer):
    """
    Prototype-based explanations for face recognition
    
    Finds and uses prototype images to explain model decisions through
    similarity and dissimilarity to known examples.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda',
                 embedding_dim: int = 512):
        """
        Initialize prototype explainer
        
        Args:
            model: Face recognition model
            device: Device to run on
            embedding_dim: Dimension of face embeddings
        """
        super().__init__(model, device)
        self.embedding_dim = embedding_dim
        
        # Storage for prototypes
        self.prototype_embeddings = None
        self.prototype_images = None
        self.prototype_identities = None
        self.prototype_metadata = None
        
        # Identity prototypes (one per identity)
        self.identity_prototypes = {}
        
    def build_prototype_database(self,
                                images: torch.Tensor,
                                identities: torch.Tensor,
                                metadata: Optional[Dict] = None,
                                n_prototypes_per_identity: int = 5) -> Dict[str, Any]:
        """
        Build database of prototype embeddings from training data
        
        Args:
            images: Training images [N, 3, H, W]
            identities: Identity labels [N]
            metadata: Optional metadata for each image
            n_prototypes_per_identity: Number of prototypes per identity
            
        Returns:
            Information about built prototype database
        """
        print("Building prototype database...")
        
        self.model.eval()
        all_embeddings = []
        all_images = []
        all_identities = []
        all_metadata = []
        
        # Extract embeddings for all images
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size].to(self.device)
                embeddings = self.model.get_embeddings(batch_images)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_images.append(batch_images.cpu().numpy())
                all_identities.extend(identities[i:i+batch_size].tolist())
                
                if metadata is not None:
                    all_metadata.extend(metadata[i:i+batch_size])
        
        # Combine all embeddings
        self.prototype_embeddings = np.vstack(all_embeddings)
        self.prototype_images = np.vstack(all_images)
        self.prototype_identities = np.array(all_identities)
        self.prototype_metadata = all_metadata if metadata else None
        
        # Build identity-specific prototypes
        unique_identities = np.unique(self.prototype_identities)
        
        for identity in unique_identities:
            identity_mask = self.prototype_identities == identity
            identity_embeddings = self.prototype_embeddings[identity_mask]
            identity_images = self.prototype_images[identity_mask]
            
            if len(identity_embeddings) <= n_prototypes_per_identity:
                # Use all if we have few examples
                prototype_indices = list(range(len(identity_embeddings)))
            else:
                # Use k-means to find representative examples
                kmeans = KMeans(n_clusters=n_prototypes_per_identity, random_state=42)
                cluster_labels = kmeans.fit_predict(identity_embeddings)
                
                # Find closest example to each cluster center
                prototype_indices = []
                for cluster_id in range(n_prototypes_per_identity):
                    cluster_mask = cluster_labels == cluster_id
                    if cluster_mask.sum() > 0:
                        cluster_embeddings = identity_embeddings[cluster_mask]
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        
                        # Find closest to center
                        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                        closest_idx = np.argmin(distances)
                        
                        # Convert back to original index
                        cluster_indices = np.where(cluster_mask)[0]
                        prototype_indices.append(cluster_indices[closest_idx])
            
            self.identity_prototypes[identity] = {
                'embeddings': identity_embeddings[prototype_indices],
                'images': identity_images[prototype_indices],
                'indices': prototype_indices
            }
        
        database_info = {
            'total_prototypes': len(self.prototype_embeddings),
            'unique_identities': len(unique_identities),
            'prototypes_per_identity': n_prototypes_per_identity,
            'embedding_dim': self.embedding_dim
        }
        
        print(f"Built prototype database: {database_info}")
        return database_info
    
    def find_nearest_prototypes(self,
                              query_embedding: np.ndarray,
                              k: int = 10,
                              exclude_identity: Optional[int] = None) -> List[Dict]:
        """
        Find k nearest prototype embeddings to query
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            k: Number of nearest prototypes to return
            exclude_identity: Identity to exclude from search
            
        Returns:
            List of prototype information dictionaries
        """
        if self.prototype_embeddings is None:
            raise ValueError("Prototype database not built. Call build_prototype_database() first.")
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.prototype_embeddings)[0]
        
        # Filter out excluded identity if specified
        if exclude_identity is not None:
            exclude_mask = self.prototype_identities != exclude_identity
            similarities = similarities * exclude_mask
        
        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        nearest_prototypes = []
        for idx in top_indices:
            prototype_info = {
                'index': idx,
                'similarity': similarities[idx],
                'identity': self.prototype_identities[idx],
                'embedding': self.prototype_embeddings[idx],
                'image': self.prototype_images[idx]
            }
            
            if self.prototype_metadata is not None:
                prototype_info['metadata'] = self.prototype_metadata[idx]
            
            nearest_prototypes.append(prototype_info)
        
        return nearest_prototypes
    
    def find_dissimilar_prototypes(self,
                                 query_embedding: np.ndarray,
                                 k: int = 5) -> List[Dict]:
        """
        Find k most dissimilar prototype embeddings
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            k: Number of dissimilar prototypes to return
            
        Returns:
            List of dissimilar prototype information
        """
        if self.prototype_embeddings is None:
            raise ValueError("Prototype database not built.")
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.prototype_embeddings)[0]
        
        # Get bottom k (most dissimilar)
        bottom_indices = np.argsort(similarities)[:k]
        
        dissimilar_prototypes = []
        for idx in bottom_indices:
            prototype_info = {
                'index': idx,
                'similarity': similarities[idx],
                'identity': self.prototype_identities[idx],
                'embedding': self.prototype_embeddings[idx],
                'image': self.prototype_images[idx]
            }
            
            if self.prototype_metadata is not None:
                prototype_info['metadata'] = self.prototype_metadata[idx]
            
            dissimilar_prototypes.append(prototype_info)
        
        return dissimilar_prototypes
    
    def explain_verification_with_prototypes(self,
                                           image1: torch.Tensor,
                                           image2: torch.Tensor,
                                           threshold: float = 0.5,
                                           k_similar: int = 5) -> Dict[str, Any]:
        """
        Explain verification decision using prototype examples
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Verification threshold
            k_similar: Number of similar prototypes to show
            
        Returns:
            Prototype-based verification explanation
        """
        # Get embeddings
        with torch.no_grad():
            emb1 = self.model.get_embeddings(image1.unsqueeze(0).to(self.device)).cpu().numpy()[0]
            emb2 = self.model.get_embeddings(image2.unsqueeze(0).to(self.device)).cpu().numpy()[0]
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        is_match = similarity > threshold
        
        # Find prototypes similar to each image
        similar_to_img1 = self.find_nearest_prototypes(emb1, k=k_similar)
        similar_to_img2 = self.find_nearest_prototypes(emb2, k=k_similar)
        
        # Find prototypes similar to the pair (average embedding)
        avg_embedding = (emb1 + emb2) / 2
        similar_to_pair = self.find_nearest_prototypes(avg_embedding, k=k_similar)
        
        # Generate explanation
        explanation = self._generate_prototype_verification_explanation(
            similarity, threshold, similar_to_img1, similar_to_img2, similar_to_pair
        )
        
        return {
            'similarity': similarity,
            'is_match': is_match,
            'threshold': threshold,
            'similar_to_image1': similar_to_img1,
            'similar_to_image2': similar_to_img2,
            'similar_to_pair': similar_to_pair,
            'explanation': explanation
        }
    
    def explain_identity_with_prototypes(self,
                                       image: torch.Tensor,
                                       predicted_identity: Optional[int] = None,
                                       k_similar: int = 5,
                                       k_dissimilar: int = 3) -> Dict[str, Any]:
        """
        Explain identity prediction using prototype examples
        
        Args:
            image: Input face image
            predicted_identity: Predicted identity (if known)
            k_similar: Number of similar prototypes
            k_dissimilar: Number of dissimilar prototypes
            
        Returns:
            Prototype-based identity explanation
        """
        # Get embedding
        with torch.no_grad():
            embedding = self.model.get_embeddings(image.unsqueeze(0).to(self.device)).cpu().numpy()[0]
        
        # Find similar and dissimilar prototypes
        similar_prototypes = self.find_nearest_prototypes(embedding, k=k_similar)
        dissimilar_prototypes = self.find_dissimilar_prototypes(embedding, k=k_dissimilar)
        
        # If predicted identity is known, find prototypes from that identity
        identity_prototypes = None
        if predicted_identity is not None and predicted_identity in self.identity_prototypes:
            identity_data = self.identity_prototypes[predicted_identity]
            identity_prototypes = []
            
            for i in range(len(identity_data['embeddings'])):
                similarity = cosine_similarity([embedding], [identity_data['embeddings'][i]])[0][0]
                identity_prototypes.append({
                    'similarity': similarity,
                    'embedding': identity_data['embeddings'][i],
                    'image': identity_data['images'][i],
                    'identity': predicted_identity
                })
        
        explanation = self._generate_prototype_identity_explanation(
            similar_prototypes, dissimilar_prototypes, identity_prototypes
        )
        
        return {
            'embedding': embedding,
            'similar_prototypes': similar_prototypes,
            'dissimilar_prototypes': dissimilar_prototypes,
            'identity_prototypes': identity_prototypes,
            'explanation': explanation
        }
    
    def _generate_prototype_verification_explanation(self,
                                                   similarity: float,
                                                   threshold: float,
                                                   similar_to_img1: List[Dict],
                                                   similar_to_img2: List[Dict],
                                                   similar_to_pair: List[Dict]) -> str:
        """Generate textual explanation for prototype-based verification"""
        decision = "Match" if similarity > threshold else "No match"
        
        explanation_parts = [
            f"{decision} (similarity: {similarity:.3f}, threshold: {threshold:.3f})"
        ]
        
        if similar_to_img1:
            top_sim1 = similar_to_img1[0]
            explanation_parts.append(
                f"Image 1 most similar to identity {top_sim1['identity']} "
                f"(similarity: {top_sim1['similarity']:.3f})"
            )
        
        if similar_to_img2:
            top_sim2 = similar_to_img2[0]
            explanation_parts.append(
                f"Image 2 most similar to identity {top_sim2['identity']} "
                f"(similarity: {top_sim2['similarity']:.3f})"
            )
        
        return ". ".join(explanation_parts) + "."
    
    def _generate_prototype_identity_explanation(self,
                                               similar_prototypes: List[Dict],
                                               dissimilar_prototypes: List[Dict],
                                               identity_prototypes: Optional[List[Dict]]) -> str:
        """Generate textual explanation for prototype-based identity prediction"""
        explanation_parts = []
        
        if similar_prototypes:
            top_similar = similar_prototypes[0]
            explanation_parts.append(
                f"Most similar to identity {top_similar['identity']} "
                f"(similarity: {top_similar['similarity']:.3f})"
            )
        
        if identity_prototypes:
            avg_identity_sim = np.mean([p['similarity'] for p in identity_prototypes])
            explanation_parts.append(
                f"Average similarity to predicted identity prototypes: {avg_identity_sim:.3f}"
            )
        
        if dissimilar_prototypes:
            most_dissimilar = dissimilar_prototypes[0]
            explanation_parts.append(
                f"Least similar to identity {most_dissimilar['identity']} "
                f"(similarity: {most_dissimilar['similarity']:.3f})"
            )
        
        return ". ".join(explanation_parts) + "."
    
    def explain(self, 
                image: torch.Tensor,
                task_type: str = 'identity',
                **kwargs) -> Dict[str, Any]:
        """
        Generate prototype-based explanation
        
        Args:
            image: Input image tensor
            task_type: 'identity' or 'verification'
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing prototype explanations
        """
        if task_type == 'identity':
            return self.explain_identity_with_prototypes(image, **kwargs)
        elif task_type == 'verification':
            # Need second image for verification
            if 'image2' not in kwargs:
                raise ValueError("Second image required for verification explanation")
            return self.explain_verification_with_prototypes(image, kwargs['image2'], **kwargs)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def save_prototype_database(self, save_path: str):
        """Save prototype database to disk"""
        database = {
            'prototype_embeddings': self.prototype_embeddings,
            'prototype_images': self.prototype_images,
            'prototype_identities': self.prototype_identities,
            'prototype_metadata': self.prototype_metadata,
            'identity_prototypes': self.identity_prototypes
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(database, f)
        
        print(f"Saved prototype database to {save_path}")
    
    def load_prototype_database(self, load_path: str):
        """Load prototype database from disk"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Prototype database not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            database = pickle.load(f)
        
        self.prototype_embeddings = database['prototype_embeddings']
        self.prototype_images = database['prototype_images'] 
        self.prototype_identities = database['prototype_identities']
        self.prototype_metadata = database.get('prototype_metadata')
        self.identity_prototypes = database.get('identity_prototypes', {})
        
        print(f"Loaded prototype database from {load_path}")
    
    def visualize_prototypes(self, 
                           prototypes: List[Dict], 
                           title: str = "Prototypes") -> np.ndarray:
        """
        Create visualization of prototype images
        
        Args:
            prototypes: List of prototype dictionaries
            title: Title for the visualization
            
        Returns:
            Combined visualization image
        """
        if not prototypes:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert prototype images to displayable format
        prototype_images = []
        for proto in prototypes:
            img = tensor_to_image(torch.from_numpy(proto['image']))
            img = (img * 255).astype(np.uint8)
            
            # Add similarity text
            similarity = proto.get('similarity', 0.0)
            identity = proto.get('identity', 'Unknown')
            
            # Add text overlay
            img_with_text = img.copy()
            cv2.putText(img_with_text, f"ID:{identity}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img_with_text, f"Sim:{similarity:.3f}", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            prototype_images.append(img_with_text)
        
        # Arrange in grid
        n_images = len(prototype_images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        rows = []
        for i in range(0, n_images, grid_size):
            row_images = prototype_images[i:i+grid_size]
            
            # Pad row if needed
            while len(row_images) < grid_size:
                row_images.append(np.zeros_like(prototype_images[0]))
            
            row = np.hstack(row_images)
            rows.append(row)
        
        # Combine rows
        if rows:
            combined = np.vstack(rows)
        else:
            combined = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return combined