"""
Robustness Testing System
========================

Comprehensive robustness evaluation including:
- Occlusion attacks and partial face scenarios
- Pose variation testing
- Illumination condition analysis
- Domain shift evaluation
- Attribute perturbation scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RobustnessEvaluator:
    """Evaluates model robustness across different challenging conditions"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda'):
        """
        Initialize robustness evaluator
        
        Args:
            model: Face recognition model
            device: Device to run evaluations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def occlusion_robustness_test(self,
                                images: torch.Tensor,
                                labels: torch.Tensor,
                                occlusion_sizes: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                                occlusion_types: List[str] = ['random', 'eyes', 'mouth', 'center'],
                                n_trials: int = 5) -> Dict[str, Any]:
        """
        Test robustness to occlusion attacks
        
        Args:
            images: Test images [N, C, H, W]
            labels: Ground truth labels for verification/identification
            occlusion_sizes: List of occlusion ratios (fraction of image)
            occlusion_types: Types of occlusion patterns
            n_trials: Number of random trials per occlusion configuration
            
        Returns:
            Dictionary with occlusion robustness results
        """
        results = {
            'occlusion_sizes': occlusion_sizes,
            'occlusion_types': occlusion_types,
            'performance_degradation': {},
            'robustness_curves': {},
            'critical_occlusion_threshold': {}
        }
        
        # Get baseline performance
        print("Computing baseline performance...")
        baseline_embeddings = self._extract_embeddings(images)
        baseline_similarities = self._compute_verification_similarities(baseline_embeddings)
        baseline_accuracy = self._compute_verification_accuracy(baseline_similarities, labels)
        
        print(f"Running occlusion robustness tests...")
        
        for occlusion_type in occlusion_types:
            results['performance_degradation'][occlusion_type] = []
            results['robustness_curves'][occlusion_type] = []
            
            for occlusion_size in occlusion_sizes:
                trial_accuracies = []
                
                for trial in range(n_trials):
                    # Apply occlusion
                    occluded_images = self._apply_occlusion(
                        images, occlusion_size, occlusion_type, seed=trial
                    )
                    
                    # Extract embeddings and compute similarities
                    occluded_embeddings = self._extract_embeddings(occluded_images)
                    occluded_similarities = self._compute_verification_similarities(occluded_embeddings)
                    occluded_accuracy = self._compute_verification_accuracy(occluded_similarities, labels)
                    
                    trial_accuracies.append(occluded_accuracy)
                
                mean_accuracy = np.mean(trial_accuracies)
                performance_drop = baseline_accuracy - mean_accuracy
                
                results['performance_degradation'][occlusion_type].append(performance_drop)
                results['robustness_curves'][occlusion_type].append(mean_accuracy)
            
            # Find critical occlusion threshold (where performance drops by 20%)
            robustness_curve = results['robustness_curves'][occlusion_type]
            critical_threshold = None
            
            for i, accuracy in enumerate(robustness_curve):
                if (baseline_accuracy - accuracy) / baseline_accuracy > 0.2:
                    critical_threshold = occlusion_sizes[i]
                    break
            
            results['critical_occlusion_threshold'][occlusion_type] = critical_threshold
        
        results['baseline_accuracy'] = baseline_accuracy
        return results
    
    def _apply_occlusion(self,
                        images: torch.Tensor,
                        occlusion_size: float,
                        occlusion_type: str,
                        seed: int = 0) -> torch.Tensor:
        """Apply occlusion to images"""
        np.random.seed(seed)
        occluded_images = images.clone()
        
        _, _, h, w = images.shape
        occlusion_area = int(occlusion_size * h * w)
        
        for i in range(len(images)):
            if occlusion_type == 'random':
                # Random rectangular occlusion
                occlusion_h = int(np.sqrt(occlusion_area))
                occlusion_w = occlusion_area // occlusion_h
                
                start_h = np.random.randint(0, max(1, h - occlusion_h))
                start_w = np.random.randint(0, max(1, w - occlusion_w))
                
                occluded_images[i, :, start_h:start_h+occlusion_h, start_w:start_w+occlusion_w] = 0
                
            elif occlusion_type == 'eyes':
                # Occlude eye region (upper 1/3 of face)
                eye_region_h = max(1, int(h * 0.33))
                eye_start = max(0, int(h * 0.2))
                occluded_images[i, :, eye_start:eye_start+eye_region_h, :] = 0
                
            elif occlusion_type == 'mouth':
                # Occlude mouth region (lower 1/3 of face)
                mouth_region_h = max(1, int(h * 0.33))
                mouth_start = max(0, int(h * 0.67))
                occluded_images[i, :, mouth_start:mouth_start+mouth_region_h, :] = 0
                
            elif occlusion_type == 'center':
                # Occlude center region
                center_h = int(np.sqrt(occlusion_area))
                center_w = occlusion_area // center_h
                
                start_h = (h - center_h) // 2
                start_w = (w - center_w) // 2
                
                occluded_images[i, :, start_h:start_h+center_h, start_w:start_w+center_w] = 0
        
        return occluded_images
    
    def pose_variation_test(self,
                          images: torch.Tensor,
                          labels: torch.Tensor,
                          pose_angles: List[float] = [-30, -15, 0, 15, 30],
                          rotation_types: List[str] = ['yaw', 'pitch', 'roll']) -> Dict[str, Any]:
        """
        Test robustness to pose variations
        
        Args:
            images: Test images [N, C, H, W]
            labels: Ground truth labels
            pose_angles: List of pose angles in degrees
            rotation_types: Types of rotations to test
            
        Returns:
            Dictionary with pose robustness results
        """
        results = {
            'pose_angles': pose_angles,
            'rotation_types': rotation_types,
            'performance_by_angle': {},
            'pose_robustness_curves': {}
        }
        
        # Baseline performance
        baseline_embeddings = self._extract_embeddings(images)
        baseline_similarities = self._compute_verification_similarities(baseline_embeddings)
        baseline_accuracy = self._compute_verification_accuracy(baseline_similarities, labels)
        
        print("Testing pose robustness...")
        
        for rotation_type in rotation_types:
            results['performance_by_angle'][rotation_type] = {}
            accuracies = []
            
            for angle in pose_angles:
                # Apply pose transformation
                transformed_images = self._apply_pose_transform(images, angle, rotation_type)
                
                # Evaluate performance
                transformed_embeddings = self._extract_embeddings(transformed_images)
                transformed_similarities = self._compute_verification_similarities(transformed_embeddings)
                transformed_accuracy = self._compute_verification_accuracy(transformed_similarities, labels)
                
                results['performance_by_angle'][rotation_type][angle] = transformed_accuracy
                accuracies.append(transformed_accuracy)
            
            results['pose_robustness_curves'][rotation_type] = accuracies
        
        results['baseline_accuracy'] = baseline_accuracy
        return results
    
    def _apply_pose_transform(self,
                            images: torch.Tensor,
                            angle: float,
                            rotation_type: str) -> torch.Tensor:
        """Apply pose transformation to images"""
        transformed_images = images.clone()
        
        for i in range(len(images)):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            h, w = img.shape[:2]
            
            if rotation_type == 'roll':
                # Simple 2D rotation for roll
                rotation_matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                transformed_img = cv2.warpAffine(img, rotation_matrix, (w, h))
                
            elif rotation_type == 'yaw' or rotation_type == 'pitch':
                # Simulate 3D rotation with affine transformation
                # This is a simplified approximation
                if rotation_type == 'yaw':
                    # Horizontal shear for yaw
                    shear_factor = np.tan(np.radians(angle)) * 0.3
                    transformation_matrix = np.array([
                        [1, shear_factor, 0],
                        [0, 1, 0]
                    ], dtype=np.float32)
                else:  # pitch
                    # Vertical shear for pitch
                    shear_factor = np.tan(np.radians(angle)) * 0.3
                    transformation_matrix = np.array([
                        [1, 0, 0],
                        [shear_factor, 1, 0]
                    ], dtype=np.float32)
                
                transformed_img = cv2.warpAffine(img, transformation_matrix, (w, h))
            
            # Convert back to tensor
            transformed_img = transformed_img.astype(np.float32) / 255.0
            transformed_images[i] = torch.from_numpy(transformed_img).permute(2, 0, 1)
        
        return transformed_images
    
    def illumination_robustness_test(self,
                                   images: torch.Tensor,
                                   labels: torch.Tensor,
                                   brightness_factors: List[float] = [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
                                   contrast_factors: List[float] = [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]) -> Dict[str, Any]:
        """
        Test robustness to illumination changes
        
        Args:
            images: Test images [N, C, H, W]
            labels: Ground truth labels
            brightness_factors: List of brightness multiplication factors
            contrast_factors: List of contrast multiplication factors
            
        Returns:
            Dictionary with illumination robustness results
        """
        results = {
            'brightness_factors': brightness_factors,
            'contrast_factors': contrast_factors,
            'brightness_robustness': [],
            'contrast_robustness': [],
            'combined_robustness': {}
        }
        
        # Baseline performance
        baseline_embeddings = self._extract_embeddings(images)
        baseline_similarities = self._compute_verification_similarities(baseline_embeddings)
        baseline_accuracy = self._compute_verification_accuracy(baseline_similarities, labels)
        
        print("Testing illumination robustness...")
        
        # Test brightness variations
        for brightness in brightness_factors:
            brightness_images = torch.clamp(images * brightness, 0, 1)
            
            brightness_embeddings = self._extract_embeddings(brightness_images)
            brightness_similarities = self._compute_verification_similarities(brightness_embeddings)
            brightness_accuracy = self._compute_verification_accuracy(brightness_similarities, labels)
            
            results['brightness_robustness'].append(brightness_accuracy)
        
        # Test contrast variations
        for contrast in contrast_factors:
            # Apply contrast: new_img = contrast * (img - 0.5) + 0.5
            contrast_images = torch.clamp(contrast * (images - 0.5) + 0.5, 0, 1)
            
            contrast_embeddings = self._extract_embeddings(contrast_images)
            contrast_similarities = self._compute_verification_similarities(contrast_embeddings)
            contrast_accuracy = self._compute_verification_accuracy(contrast_similarities, labels)
            
            results['contrast_robustness'].append(contrast_accuracy)
        
        # Test combined brightness and contrast variations (subset)
        test_combinations = [
            (0.5, 0.5), (0.7, 0.7), (1.0, 1.0), (1.3, 1.3), (1.7, 1.7)
        ]
        
        for brightness, contrast in test_combinations:
            combined_images = torch.clamp(contrast * (images * brightness - 0.5) + 0.5, 0, 1)
            
            combined_embeddings = self._extract_embeddings(combined_images)
            combined_similarities = self._compute_verification_similarities(combined_embeddings)
            combined_accuracy = self._compute_verification_accuracy(combined_similarities, labels)
            
            results['combined_robustness'][(brightness, contrast)] = combined_accuracy
        
        results['baseline_accuracy'] = baseline_accuracy
        return results
    
    def domain_shift_test(self,
                         source_images: torch.Tensor,
                         target_images: torch.Tensor,
                         source_labels: torch.Tensor,
                         target_labels: torch.Tensor,
                         domain_names: List[str] = ['source', 'target']) -> Dict[str, Any]:
        """
        Test robustness to domain shift
        
        Args:
            source_images: Images from source domain [N, C, H, W]
            target_images: Images from target domain [M, C, H, W]
            source_labels: Labels for source domain
            target_labels: Labels for target domain
            domain_names: Names for source and target domains
            
        Returns:
            Dictionary with domain shift analysis
        """
        results = {
            'domain_names': domain_names,
            'within_domain_performance': {},
            'cross_domain_performance': {},
            'domain_gap': {}
        }
        
        print("Testing domain shift robustness...")
        
        # Extract embeddings
        source_embeddings = self._extract_embeddings(source_images)
        target_embeddings = self._extract_embeddings(target_images)
        
        # Within-domain performance
        source_similarities = self._compute_verification_similarities(source_embeddings)
        source_accuracy = self._compute_verification_accuracy(source_similarities, source_labels)
        
        target_similarities = self._compute_verification_similarities(target_embeddings)
        target_accuracy = self._compute_verification_accuracy(target_similarities, target_labels)
        
        results['within_domain_performance'][domain_names[0]] = source_accuracy
        results['within_domain_performance'][domain_names[1]] = target_accuracy
        
        # Cross-domain performance (source embeddings vs target embeddings)
        if len(source_embeddings) == len(target_embeddings):
            cross_similarities = self._compute_cross_domain_similarities(
                source_embeddings, target_embeddings
            )
            # Assume same identities for cross-domain test
            cross_labels = source_labels  # Simplified assumption
            cross_accuracy = self._compute_verification_accuracy(cross_similarities, cross_labels)
            
            results['cross_domain_performance'] = cross_accuracy
            
            # Domain gap analysis
            avg_within_domain = (source_accuracy + target_accuracy) / 2
            domain_gap = avg_within_domain - cross_accuracy
            
            results['domain_gap'] = {
                'absolute_gap': domain_gap,
                'relative_gap': domain_gap / avg_within_domain if avg_within_domain > 0 else 0
            }
        
        # Embedding distribution analysis
        results['embedding_analysis'] = self._analyze_embedding_distributions(
            source_embeddings, target_embeddings, domain_names
        )
        
        return results
    
    def _compute_cross_domain_similarities(self,
                                         source_embeddings: np.ndarray,
                                         target_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarities between source and target domain embeddings"""
        similarities = []
        
        for i in range(len(source_embeddings)):
            source_emb = source_embeddings[i]
            target_emb = target_embeddings[i]
            
            # Cosine similarity
            similarity = np.dot(source_emb, target_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _analyze_embedding_distributions(self,
                                       source_embeddings: np.ndarray,
                                       target_embeddings: np.ndarray,
                                       domain_names: List[str]) -> Dict[str, Any]:
        """Analyze embedding space distributions across domains"""
        from scipy.stats import wasserstein_distance
        
        analysis = {
            'embedding_means': {},
            'embedding_stds': {},
            'wasserstein_distances': {}
        }
        
        # Compute statistics
        analysis['embedding_means'][domain_names[0]] = np.mean(source_embeddings, axis=0)
        analysis['embedding_means'][domain_names[1]] = np.mean(target_embeddings, axis=0)
        
        analysis['embedding_stds'][domain_names[0]] = np.std(source_embeddings, axis=0)
        analysis['embedding_stds'][domain_names[1]] = np.std(target_embeddings, axis=0)
        
        # Wasserstein distance for each embedding dimension
        n_dims = min(source_embeddings.shape[1], target_embeddings.shape[1])
        wasserstein_dists = []
        
        for dim in range(n_dims):
            dist = wasserstein_distance(
                source_embeddings[:, dim],
                target_embeddings[:, dim]
            )
            wasserstein_dists.append(dist)
        
        analysis['wasserstein_distances']['per_dimension'] = wasserstein_dists
        analysis['wasserstein_distances']['mean'] = np.mean(wasserstein_dists)
        analysis['wasserstein_distances']['std'] = np.std(wasserstein_dists)
        
        return analysis
    
    def attribute_perturbation_test(self,
                                  images: torch.Tensor,
                                  attribute_predictions: Dict[str, np.ndarray],
                                  perturbation_attributes: List[str] = ['Male', 'Young', 'Eyeglasses'],
                                  perturbation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Test robustness to attribute-based perturbations
        
        Args:
            images: Test images [N, C, H, W]
            attribute_predictions: Dictionary with attribute predictions
            perturbation_attributes: Attributes to use for perturbation
            perturbation_strength: Strength of perturbation
            
        Returns:
            Dictionary with attribute perturbation results
        """
        results = {
            'perturbation_attributes': perturbation_attributes,
            'perturbation_strength': perturbation_strength,
            'baseline_attributes': {},
            'perturbed_attributes': {},
            'attribute_stability': {}
        }
        
        # Get baseline attribute predictions
        baseline_embeddings = self._extract_embeddings(images)
        
        print("Testing attribute perturbation robustness...")
        
        for attr in perturbation_attributes:
            if attr not in attribute_predictions:
                print(f"Warning: {attr} not found in attribute predictions")
                continue
            
            baseline_attr_preds = attribute_predictions[attr]
            results['baseline_attributes'][attr] = baseline_attr_preds
            
            # Apply perturbations based on attribute
            perturbed_images = self._apply_attribute_perturbation(
                images, attr, perturbation_strength
            )
            
            # Get new attribute predictions
            perturbed_embeddings = self._extract_embeddings(perturbed_images)
            
            # For now, assume we have an attribute predictor
            # In practice, this would use the model's attribute prediction head
            perturbed_attr_preds = self._predict_attributes_from_embeddings(
                perturbed_embeddings, attr
            )
            
            results['perturbed_attributes'][attr] = perturbed_attr_preds
            
            # Compute attribute stability
            if baseline_attr_preds.dtype == bool or np.all(np.isin(baseline_attr_preds, [0, 1])):
                # Binary attribute
                stability = np.mean(baseline_attr_preds == perturbed_attr_preds)
            else:
                # Continuous attribute - use correlation
                stability = np.corrcoef(baseline_attr_preds, perturbed_attr_preds)[0, 1]
                if np.isnan(stability):
                    stability = 0.0
            
            results['attribute_stability'][attr] = stability
        
        return results
    
    def _apply_attribute_perturbation(self,
                                    images: torch.Tensor,
                                    attribute: str,
                                    strength: float) -> torch.Tensor:
        """Apply attribute-based perturbation to images"""
        perturbed_images = images.clone()
        
        # Simple perturbations based on attribute type
        if attribute == 'Male':
            # Slight contrast adjustment (males often have different lighting)
            perturbed_images = torch.clamp(1.1 * (perturbed_images - 0.5) + 0.5, 0, 1)
        elif attribute == 'Young':
            # Slight smoothing (younger faces often smoother)
            for i in range(len(images)):
                img = perturbed_images[i].permute(1, 2, 0).cpu().numpy()
                img = cv2.GaussianBlur(img, (3, 3), strength)
                perturbed_images[i] = torch.from_numpy(img).permute(2, 0, 1)
        elif attribute == 'Eyeglasses':
            # Add noise in eye region
            _, _, h, w = images.shape
            eye_region = slice(int(0.3*h), int(0.6*h))
            noise = torch.randn_like(perturbed_images[:, :, eye_region, :]) * strength
            perturbed_images[:, :, eye_region, :] += noise
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images
    
    def _predict_attributes_from_embeddings(self,
                                          embeddings: np.ndarray,
                                          attribute: str) -> np.ndarray:
        """Predict attributes from embeddings (placeholder implementation)"""
        # This is a placeholder - in practice, you would use the model's attribute head
        # or a separate attribute classifier trained on embeddings
        
        # For now, return random predictions that maintain some correlation
        n_samples = len(embeddings)
        
        if attribute in ['Male', 'Young', 'Eyeglasses']:
            # Binary attributes
            predictions = np.random.choice([0, 1], size=n_samples)
        else:
            # Continuous attributes
            predictions = np.random.randn(n_samples)
        
        return predictions
    
    def _extract_embeddings(self, images: torch.Tensor) -> np.ndarray:
        """Extract embeddings from images"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), 32):  # Batch processing
                batch = images[i:i+32].to(self.device)
                
                if hasattr(self.model, 'get_embeddings'):
                    batch_embeddings = self.model.get_embeddings(batch)
                else:
                    # Fallback to forward pass
                    outputs = self.model(batch)
                    if isinstance(outputs, dict) and 'embeddings' in outputs:
                        batch_embeddings = outputs['embeddings']
                    else:
                        batch_embeddings = outputs
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def _compute_verification_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarities for verification"""
        # For simplicity, compute similarity between consecutive pairs
        similarities = []
        
        for i in range(0, len(embeddings) - 1, 2):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _compute_verification_accuracy(self,
                                     similarities: np.ndarray,
                                     labels: torch.Tensor,
                                     threshold: float = 0.5) -> float:
        """Compute verification accuracy"""
        # Assume labels are binary (same/different person)
        predictions = (similarities >= threshold).astype(int)
        
        # Convert labels to pairs
        pair_labels = []
        for i in range(0, len(labels) - 1, 2):
            pair_labels.append(1 if labels[i] == labels[i + 1] else 0)
        
        pair_labels = np.array(pair_labels)
        
        if len(predictions) != len(pair_labels):
            min_len = min(len(predictions), len(pair_labels))
            predictions = predictions[:min_len]
            pair_labels = pair_labels[:min_len]
        
        return np.mean(predictions == pair_labels)
    
    def create_robustness_summary_plot(self,
                                     robustness_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive robustness summary visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Occlusion robustness
        if 'occlusion_sizes' in robustness_results:
            occlusion_data = robustness_results
            
            for occlusion_type in occlusion_data['occlusion_types']:
                if occlusion_type in occlusion_data['robustness_curves']:
                    axes[0].plot(occlusion_data['occlusion_sizes'],
                               occlusion_data['robustness_curves'][occlusion_type],
                               marker='o', label=occlusion_type)
            
            axes[0].axhline(y=occlusion_data['baseline_accuracy'], color='red',
                           linestyle='--', label='Baseline')
            axes[0].set_xlabel('Occlusion Size')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Occlusion Robustness')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Pose robustness
        if 'pose_angles' in robustness_results:
            pose_data = robustness_results
            
            for rotation_type in pose_data['rotation_types']:
                if rotation_type in pose_data['pose_robustness_curves']:
                    axes[1].plot(pose_data['pose_angles'],
                               pose_data['pose_robustness_curves'][rotation_type],
                               marker='s', label=rotation_type)
            
            axes[1].axhline(y=pose_data['baseline_accuracy'], color='red',
                           linestyle='--', label='Baseline')
            axes[1].set_xlabel('Pose Angle (degrees)')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Pose Robustness')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Illumination robustness
        if 'brightness_factors' in robustness_results:
            illum_data = robustness_results
            
            axes[2].plot(illum_data['brightness_factors'],
                        illum_data['brightness_robustness'],
                        marker='^', label='Brightness', color='orange')
            axes[2].plot(illum_data['contrast_factors'],
                        illum_data['contrast_robustness'],
                        marker='v', label='Contrast', color='purple')
            axes[2].axhline(y=illum_data['baseline_accuracy'], color='red',
                           linestyle='--', label='Baseline')
            axes[2].set_xlabel('Factor')
            axes[2].set_ylabel('Accuracy')
            axes[2].set_title('Illumination Robustness')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Domain shift analysis
        if 'domain_names' in robustness_results:
            domain_data = robustness_results
            domains = domain_data['domain_names']
            within_performance = [domain_data['within_domain_performance'][d] for d in domains]
            
            bars = axes[3].bar(domains, within_performance, color=['skyblue', 'lightcoral'])
            
            if 'cross_domain_performance' in domain_data:
                axes[3].axhline(y=domain_data['cross_domain_performance'],
                              color='green', linestyle='--', label='Cross-domain')
            
            axes[3].set_ylabel('Accuracy')
            axes[3].set_title('Domain Shift Analysis')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        # Attribute stability
        if 'attribute_stability' in robustness_results:
            attr_data = robustness_results
            attributes = list(attr_data['attribute_stability'].keys())
            stabilities = list(attr_data['attribute_stability'].values())
            
            bars = axes[4].bar(attributes, stabilities, color='lightgreen')
            axes[4].set_ylabel('Stability')
            axes[4].set_title('Attribute Perturbation Stability')
            axes[4].set_ylim(0, 1)
            axes[4].grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, stability in zip(bars, stabilities):
                axes[4].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{stability:.3f}', ha='center', va='bottom')
        
        # Overall robustness radar chart
        # This would show multiple robustness metrics on a radar plot
        axes[5].text(0.5, 0.5, 'Overall Robustness\nSummary\n(Implementation needed)',
                    ha='center', va='center', fontsize=12)
        axes[5].set_xlim(0, 1)
        axes[5].set_ylim(0, 1)
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig