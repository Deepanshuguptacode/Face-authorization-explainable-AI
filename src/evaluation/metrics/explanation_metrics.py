"""
Explanation Metrics Evaluator
============================

Comprehensive evaluation of explainability methods including:
- Sanity checks for saliency methods
- Fidelity measures (deletion/insertion metrics)
- TCAV significance tests
- Human evaluation framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import auc
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

class ExplanationEvaluator:
    """Evaluates explanation method fidelity and reliability"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 explainer_factory: Callable,
                 device: str = 'cuda'):
        """
        Initialize explanation evaluator
        
        Args:
            model: Face recognition model
            explainer_factory: Function that creates explainer instances
            device: Device to run evaluations on
        """
        self.model = model
        self.explainer_factory = explainer_factory
        self.device = device
        self.model.eval()
    
    def sanity_check_model_dependence(self,
                                    images: torch.Tensor,
                                    n_trials: int = 10,
                                    explanation_method: str = 'gradcam') -> Dict[str, Any]:
        """
        Sanity check: randomized model weights should break saliency signals
        
        This test ensures explanations actually depend on the trained model
        rather than just input image statistics.
        
        Args:
            images: Test images [N, C, H, W]
            n_trials: Number of randomization trials
            explanation_method: Explanation method to test
            
        Returns:
            Dictionary with sanity check results
        """
        results = {
            'method': explanation_method,
            'original_explanations': [],
            'randomized_explanations': [],
            'similarity_scores': [],
            'rank_correlations': []
        }
        
        # Get original explanations
        explainer = self.explainer_factory(self.model)
        
        print(f"Computing original {explanation_method} explanations...")
        for i in tqdm(range(len(images))):
            image = images[i:i+1]
            explanation = explainer.explain(image, target_type='identity')
            
            if explanation_method == 'gradcam':
                saliency = explanation['grad_cam']
            elif explanation_method == 'integrated_gradients':
                saliency = explanation['integrated_gradients']
            else:
                raise ValueError(f"Unsupported method: {explanation_method}")
            
            results['original_explanations'].append(saliency.flatten())
        
        # Test with randomized models
        print(f"Testing with {n_trials} randomized models...")
        for trial in tqdm(range(n_trials)):
            # Create model copy and randomize weights
            randomized_model = self._randomize_model_weights(self.model)
            randomized_explainer = self.explainer_factory(randomized_model)
            
            trial_explanations = []
            for i in range(len(images)):
                image = images[i:i+1]
                explanation = randomized_explainer.explain(image, target_type='identity')
                
                if explanation_method == 'gradcam':
                    saliency = explanation['grad_cam']
                elif explanation_method == 'integrated_gradients':
                    saliency = explanation['integrated_gradients']
                
                trial_explanations.append(saliency.flatten())
            
            results['randomized_explanations'].append(trial_explanations)
        
        # Compute similarity metrics
        print("Computing similarity metrics...")
        for trial_explanations in results['randomized_explanations']:
            trial_similarities = []
            trial_correlations = []
            
            for i in range(len(images)):
                orig_exp = results['original_explanations'][i]
                rand_exp = trial_explanations[i]
                
                # Cosine similarity
                similarity = np.dot(orig_exp, rand_exp) / (
                    np.linalg.norm(orig_exp) * np.linalg.norm(rand_exp) + 1e-8
                )
                trial_similarities.append(similarity)
                
                # Rank correlation
                correlation = stats.spearmanr(orig_exp, rand_exp)[0]
                if not np.isnan(correlation):
                    trial_correlations.append(correlation)
            
            results['similarity_scores'].append(np.mean(trial_similarities))
            results['rank_correlations'].append(np.mean(trial_correlations))
        
        # Statistical analysis
        results['mean_similarity'] = np.mean(results['similarity_scores'])
        results['std_similarity'] = np.std(results['similarity_scores'])
        results['mean_correlation'] = np.mean(results['rank_correlations'])
        results['std_correlation'] = np.std(results['rank_correlations'])
        
        # Test if similarities are significantly lower than random
        # (Good explanations should have low similarity with randomized models)
        results['similarity_pvalue'] = stats.ttest_1samp(
            results['similarity_scores'], 0.0
        )[1]
        
        return results
    
    def _randomize_model_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """Create a copy of model with randomized weights"""
        import copy
        
        randomized_model = copy.deepcopy(model)
        
        # Randomize all parameters
        with torch.no_grad():
            for param in randomized_model.parameters():
                param.data = torch.randn_like(param.data) * 0.01
        
        randomized_model.eval()
        return randomized_model
    
    def deletion_insertion_test(self,
                               images: torch.Tensor,
                               explanations: List[np.ndarray],
                               step_size: float = 0.1,
                               baseline_value: float = 0.0) -> Dict[str, Any]:
        """
        Deletion/Insertion test for explanation fidelity
        
        Measures how model predictions change when important pixels
        (high saliency) are deleted or unimportant pixels are inserted.
        
        Args:
            images: Test images [N, C, H, W]
            explanations: Saliency maps for each image
            step_size: Fraction of pixels to modify at each step
            baseline_value: Value to use for deleted pixels
            
        Returns:
            Dictionary with deletion/insertion curves
        """
        results = {
            'deletion_scores': [],
            'insertion_scores': [],
            'steps': np.arange(0, 1 + step_size, step_size)
        }
        
        print("Running deletion/insertion test...")
        
        for i in tqdm(range(len(images))):
            image = images[i:i+1].clone()
            saliency = explanations[i]
            
            # Get original prediction confidence
            with torch.no_grad():
                original_output = self.model(image.to(self.device))
                if 'identity_logits' in original_output:
                    original_probs = F.softmax(original_output['identity_logits'], dim=1)
                    original_confidence = torch.max(original_probs).item()
                    predicted_class = torch.argmax(original_probs, dim=1).item()
                else:
                    # Use embedding magnitude as confidence proxy
                    original_confidence = torch.norm(
                        self.model.get_embeddings(image.to(self.device))
                    ).item()
                    predicted_class = 0
            
            # Sort pixels by saliency (most important first)
            flat_saliency = saliency.flatten()
            sorted_indices = np.argsort(flat_saliency)[::-1]
            
            # Deletion test (remove most important pixels first)
            deletion_confidences = [original_confidence]
            modified_image = image.clone()
            
            for step in results['steps'][1:]:
                n_pixels = int(step * len(sorted_indices))
                pixels_to_delete = sorted_indices[:n_pixels]
                
                # Convert flat indices to 2D coordinates
                h, w = image.shape[2], image.shape[3]
                y_coords = pixels_to_delete // w
                x_coords = pixels_to_delete % w
                
                # Delete pixels (set to baseline value)
                modified_image[0, :, y_coords, x_coords] = baseline_value
                
                # Get new prediction
                with torch.no_grad():
                    output = self.model(modified_image.to(self.device))
                    if 'identity_logits' in output:
                        probs = F.softmax(output['identity_logits'], dim=1)
                        confidence = probs[0, predicted_class].item()
                    else:
                        confidence = torch.norm(
                            self.model.get_embeddings(modified_image.to(self.device))
                        ).item()
                
                deletion_confidences.append(confidence)
            
            # Insertion test (start with baseline, add most important pixels)
            insertion_confidences = []
            modified_image = torch.full_like(image, baseline_value)
            
            for step in results['steps']:
                if step == 0:
                    # Fully baseline image
                    with torch.no_grad():
                        output = self.model(modified_image.to(self.device))
                        if 'identity_logits' in output:
                            probs = F.softmax(output['identity_logits'], dim=1)
                            confidence = probs[0, predicted_class].item()
                        else:
                            confidence = torch.norm(
                                self.model.get_embeddings(modified_image.to(self.device))
                            ).item()
                    insertion_confidences.append(confidence)
                else:
                    n_pixels = int(step * len(sorted_indices))
                    pixels_to_insert = sorted_indices[:n_pixels]
                    
                    # Convert flat indices to 2D coordinates
                    y_coords = pixels_to_insert // w
                    x_coords = pixels_to_insert % w
                    
                    # Insert original pixel values
                    modified_image[0, :, y_coords, x_coords] = image[0, :, y_coords, x_coords]
                    
                    # Get new prediction
                    with torch.no_grad():
                        output = self.model(modified_image.to(self.device))
                        if 'identity_logits' in output:
                            probs = F.softmax(output['identity_logits'], dim=1)
                            confidence = probs[0, predicted_class].item()
                        else:
                            confidence = torch.norm(
                                self.model.get_embeddings(modified_image.to(self.device))
                            ).item()
                    
                    insertion_confidences.append(confidence)
            
            results['deletion_scores'].append(deletion_confidences)
            results['insertion_scores'].append(insertion_confidences)
        
        # Compute summary metrics
        deletion_curves = np.array(results['deletion_scores'])
        insertion_curves = np.array(results['insertion_scores'])
        
        results['deletion_auc'] = np.mean([
            auc(results['steps'], curve) for curve in deletion_curves
        ])
        
        results['insertion_auc'] = np.mean([
            auc(results['steps'], curve) for curve in insertion_curves
        ])
        
        results['mean_deletion_curve'] = np.mean(deletion_curves, axis=0)
        results['mean_insertion_curve'] = np.mean(insertion_curves, axis=0)
        
        return results
    
    def localization_test(self,
                         images: torch.Tensor,
                         explanations: List[np.ndarray],
                         ground_truth_masks: Optional[List[np.ndarray]] = None,
                         target_regions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test explanation localization using ground truth attention regions
        
        Args:
            images: Test images [N, C, H, W]
            explanations: Saliency maps for each image
            ground_truth_masks: Ground truth attention masks (if available)
            target_regions: Target regions to focus on (e.g., 'eyes', 'mouth')
            
        Returns:
            Dictionary with localization metrics
        """
        results = {}
        
        if ground_truth_masks is not None:
            # Compute IoU with ground truth masks
            ious = []
            for i in range(len(explanations)):
                saliency = explanations[i]
                gt_mask = ground_truth_masks[i]
                
                # Threshold saliency to create binary mask
                saliency_binary = (saliency > np.percentile(saliency, 80)).astype(int)
                gt_binary = (gt_mask > 0.5).astype(int)
                
                # Compute IoU
                intersection = np.sum(saliency_binary * gt_binary)
                union = np.sum((saliency_binary + gt_binary) > 0)
                
                if union > 0:
                    iou = intersection / union
                    ious.append(iou)
            
            results['mean_iou'] = np.mean(ious)
            results['std_iou'] = np.std(ious)
            results['individual_ious'] = ious
        
        # Face region analysis (approximate)
        face_region_scores = []
        for i in range(len(explanations)):
            saliency = explanations[i]
            h, w = saliency.shape
            
            # Define approximate face regions (center region)
            center_h_start, center_h_end = int(0.25 * h), int(0.75 * h)
            center_w_start, center_w_end = int(0.25 * w), int(0.75 * w)
            
            # Compute fraction of saliency in face region
            total_saliency = np.sum(saliency)
            face_saliency = np.sum(saliency[center_h_start:center_h_end, 
                                           center_w_start:center_w_end])
            
            if total_saliency > 0:
                face_score = face_saliency / total_saliency
                face_region_scores.append(face_score)
        
        results['mean_face_focus'] = np.mean(face_region_scores)
        results['std_face_focus'] = np.std(face_region_scores)
        
        return results
    
    def tcav_significance_test(self,
                              tcav_scores: np.ndarray,
                              n_bootstrap: int = 1000,
                              alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of TCAV scores
        
        Args:
            tcav_scores: TCAV scores for concept [N]
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
            
        Returns:
            Dictionary with significance test results
        """
        results = {
            'mean_tcav_score': np.mean(tcav_scores),
            'std_tcav_score': np.std(tcav_scores),
            'n_samples': len(tcav_scores)
        }
        
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(tcav_scores, len(tcav_scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results['confidence_interval'] = (
            np.percentile(bootstrap_means, lower_percentile),
            np.percentile(bootstrap_means, upper_percentile)
        )
        
        # Statistical significance test (H0: TCAV score = 0.5, random)
        t_stat, p_value = stats.ttest_1samp(tcav_scores, 0.5)
        
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        results['is_significant'] = p_value < alpha
        
        # Effect size (Cohen's d)
        results['effect_size'] = (np.mean(tcav_scores) - 0.5) / np.std(tcav_scores)
        
        return results
    
    def human_evaluation_framework(self,
                                 images: torch.Tensor,
                                 explanations: List[Dict],
                                 save_dir: str = "human_eval_data") -> Dict[str, Any]:
        """
        Prepare data for human evaluation of explanations
        
        Creates standardized visualization pairs for human assessment
        of explanation quality and helpfulness.
        
        Args:
            images: Test images [N, C, H, W]
            explanations: Explanation results for each image
            save_dir: Directory to save evaluation materials
            
        Returns:
            Dictionary with evaluation setup information
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        evaluation_data = {
            'image_pairs': [],
            'evaluation_questions': self._create_evaluation_questions(),
            'rating_scales': self._create_rating_scales(),
            'instructions': self._create_evaluation_instructions()
        }
        
        print(f"Preparing human evaluation data for {len(images)} images...")
        
        for i in tqdm(range(len(images))):
            image = images[i]
            explanation = explanations[i]
            
            # Create visualization pair
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Saliency map
            if 'grad_cam' in explanation:
                saliency = explanation['grad_cam']
                axes[1].imshow(saliency, cmap='hot')
                axes[1].set_title('Attention Map')
                axes[1].axis('off')
            
            # Overlay
            if 'grad_cam_overlay' in explanation:
                overlay = explanation['grad_cam_overlay']
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            save_path = os.path.join(save_dir, f'eval_pair_{i:04d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store metadata
            evaluation_data['image_pairs'].append({
                'image_id': i,
                'visualization_path': save_path,
                'explanation_metadata': {
                    'method': 'grad_cam',
                    'target_class': explanation.get('target_class', 'unknown'),
                    'confidence': explanation.get('confidence', 0.0)
                }
            })
        
        # Save evaluation protocol
        protocol_path = os.path.join(save_dir, 'evaluation_protocol.json')
        import json
        with open(protocol_path, 'w') as f:
            json.dump({
                'questions': evaluation_data['evaluation_questions'],
                'rating_scales': evaluation_data['rating_scales'],
                'instructions': evaluation_data['instructions']
            }, f, indent=2)
        
        evaluation_data['protocol_path'] = protocol_path
        evaluation_data['n_images'] = len(images)
        
        return evaluation_data
    
    def _create_evaluation_questions(self) -> List[Dict]:
        """Create standardized evaluation questions"""
        return [
            {
                'id': 'helpfulness',
                'question': 'How helpful is this explanation for understanding the model\'s decision?',
                'type': 'likert_5'
            },
            {
                'id': 'accuracy',
                'question': 'How accurately does the highlighted region correspond to important facial features?',
                'type': 'likert_5'
            },
            {
                'id': 'trust',
                'question': 'How much do you trust this explanation?',
                'type': 'likert_5'
            },
            {
                'id': 'clarity',
                'question': 'How clear and interpretable is this explanation?',
                'type': 'likert_5'
            },
            {
                'id': 'focus_region',
                'question': 'Which facial region does the explanation focus on most?',
                'type': 'multiple_choice',
                'options': ['Eyes', 'Nose', 'Mouth', 'Overall face', 'Background', 'Unclear']
            }
        ]
    
    def _create_rating_scales(self) -> Dict:
        """Create rating scale definitions"""
        return {
            'likert_5': {
                1: 'Strongly Disagree',
                2: 'Disagree', 
                3: 'Neutral',
                4: 'Agree',
                5: 'Strongly Agree'
            }
        }
    
    def _create_evaluation_instructions(self) -> str:
        """Create evaluation instructions for human evaluators"""
        return """
        EXPLANATION EVALUATION INSTRUCTIONS
        
        You will be shown pairs of images with their corresponding AI explanations.
        Each pair contains:
        1. Original face image
        2. Attention/saliency map showing which regions the AI focused on
        3. Overlay combining the original image with the attention map
        
        Please evaluate each explanation based on the provided questions.
        Consider:
        - Does the explanation help you understand why the AI made its decision?
        - Do the highlighted regions correspond to meaningful facial features?
        - Is the explanation clear and interpretable?
        - How much confidence would you have in this AI system based on this explanation?
        
        Rate each question on the provided scale.
        """
    
    def create_sanity_check_plot(self,
                               results: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization for sanity check results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Similarity scores distribution
        ax1.hist(results['similarity_scores'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(results['mean_similarity'], color='red', linestyle='--', 
                   label=f'Mean: {results["mean_similarity"]:.3f}')
        ax1.set_xlabel('Cosine Similarity with Randomized Model')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Sanity Check: {results["method"].title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rank correlation distribution
        ax2.hist(results['rank_correlations'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(results['mean_correlation'], color='red', linestyle='--',
                   label=f'Mean: {results["mean_correlation"]:.3f}')
        ax2.set_xlabel('Spearman Correlation with Randomized Model')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Rank Correlation Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_deletion_insertion_plot(self,
                                     results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Create deletion/insertion curves plot"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean curves with error bands
        steps = results['steps']
        deletion_curves = np.array(results['deletion_scores'])
        insertion_curves = np.array(results['insertion_scores'])
        
        deletion_mean = np.mean(deletion_curves, axis=0)
        deletion_std = np.std(deletion_curves, axis=0)
        
        insertion_mean = np.mean(insertion_curves, axis=0)
        insertion_std = np.std(insertion_curves, axis=0)
        
        # Deletion curve (should decrease)
        ax.plot(steps, deletion_mean, 'r-', linewidth=2, 
               label=f'Deletion (AUC: {results["deletion_auc"]:.3f})')
        ax.fill_between(steps, deletion_mean - deletion_std, deletion_mean + deletion_std,
                       alpha=0.3, color='red')
        
        # Insertion curve (should increase)
        ax.plot(steps, insertion_mean, 'b-', linewidth=2,
               label=f'Insertion (AUC: {results["insertion_auc"]:.3f})')
        ax.fill_between(steps, insertion_mean - insertion_std, insertion_mean + insertion_std,
                       alpha=0.3, color='blue')
        
        ax.set_xlabel('Fraction of Pixels Modified')
        ax.set_ylabel('Model Confidence')
        ax.set_title('Deletion/Insertion Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig