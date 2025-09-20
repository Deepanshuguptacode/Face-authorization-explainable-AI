"""
Recognition Metrics Evaluator
============================

Comprehensive evaluation of face recognition performance including:
- Verification metrics: TAR@FAR, ROC, AUC, EER
- Identification metrics: Rank-1, Rank-5 accuracy
- Demographic disaggregation
- Attribute condition analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional, Any
import torch
import warnings
warnings.filterwarnings('ignore')

class RecognitionEvaluator:
    """Evaluates face recognition performance with comprehensive metrics"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda',
                 attribute_names: Optional[List[str]] = None):
        """
        Initialize recognition evaluator
        
        Args:
            model: Face recognition model
            device: Device to run evaluations on
            attribute_names: List of attribute names for demographic analysis
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Default CelebA attribute names
        self.attribute_names = attribute_names or [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        
        # Demographics for fairness analysis
        self.demographic_attributes = {
            'gender': 'Male',
            'age': 'Young', 
            'skin_tone': 'Pale_Skin',
            'glasses': 'Eyeglasses'
        }
        
    def compute_embeddings(self, 
                          images: torch.Tensor, 
                          batch_size: int = 32) -> np.ndarray:
        """
        Compute embeddings for a batch of images
        
        Args:
            images: Input images [N, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array [N, embedding_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                emb = self.model.get_embeddings(batch)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_similarity_matrix(self, 
                                 embeddings1: np.ndarray,
                                 embeddings2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix between embeddings
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D] (if None, use embeddings1)
            
        Returns:
            Similarity matrix [N, M] or [N, N]
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
            
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        return similarity_matrix
    
    def evaluate_verification(self,
                            pair_similarities: np.ndarray,
                            pair_labels: np.ndarray,
                            demographic_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate face verification performance
        
        Args:
            pair_similarities: Similarity scores for pairs [N]
            pair_labels: True labels (1=same person, 0=different) [N]
            demographic_data: Optional demographic information for disaggregation
            
        Returns:
            Dictionary with verification metrics
        """
        results = {}
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(pair_labels, pair_similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # Compute TAR @ specific FARs
        target_fars = [0.001, 0.01, 0.1]
        tar_at_far = {}
        
        for far in target_fars:
            # Find closest FPR to target FAR
            far_idx = np.argmin(np.abs(fpr - far))
            tar_at_far[f'TAR@FAR={far}'] = tpr[far_idx]
        
        # Store main results
        results['overall'] = {
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'tar_at_far': tar_at_far,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        # Demographic disaggregation if provided
        if demographic_data is not None:
            results['demographic'] = self._evaluate_by_demographics(
                pair_similarities, pair_labels, demographic_data
            )
        
        return results
    
    def evaluate_identification(self,
                              query_embeddings: np.ndarray,
                              gallery_embeddings: np.ndarray,
                              query_identities: np.ndarray,
                              gallery_identities: np.ndarray,
                              demographic_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate face identification performance
        
        Args:
            query_embeddings: Query embeddings [N_q, D]
            gallery_embeddings: Gallery embeddings [N_g, D]
            query_identities: Query identity labels [N_q]
            gallery_identities: Gallery identity labels [N_g]
            demographic_data: Optional demographic information
            
        Returns:
            Dictionary with identification metrics
        """
        results = {}
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(
            query_embeddings, gallery_embeddings
        )
        
        # Compute ranking metrics
        ranks = [1, 5, 10, 20]
        rank_accuracies = {}
        
        for rank in ranks:
            correct = 0
            for i, query_id in enumerate(query_identities):
                # Get top-k gallery indices for this query
                top_k_indices = np.argsort(similarity_matrix[i])[::-1][:rank]
                top_k_gallery_ids = gallery_identities[top_k_indices]
                
                # Check if true identity is in top-k
                if query_id in top_k_gallery_ids:
                    correct += 1
            
            rank_accuracies[f'Rank-{rank}'] = correct / len(query_identities)
        
        results['overall'] = rank_accuracies
        
        # Demographic disaggregation if provided
        if demographic_data is not None:
            results['demographic'] = self._evaluate_identification_by_demographics(
                similarity_matrix, query_identities, gallery_identities, 
                demographic_data, ranks
            )
        
        return results
    
    def _evaluate_by_demographics(self,
                                similarities: np.ndarray,
                                labels: np.ndarray,
                                demographic_data: Dict) -> Dict[str, Any]:
        """Evaluate verification performance by demographic groups"""
        demo_results = {}
        
        for demo_name, demo_values in demographic_data.items():
            demo_results[demo_name] = {}
            
            # Get unique demographic values
            unique_values = np.unique(demo_values)
            
            for value in unique_values:
                # Filter data for this demographic group
                mask = demo_values == value
                if np.sum(mask) < 10:  # Skip groups with too few samples
                    continue
                    
                group_similarities = similarities[mask]
                group_labels = labels[mask]
                
                # Compute metrics for this group
                fpr, tpr, _ = roc_curve(group_labels, group_similarities)
                group_auc = auc(fpr, tpr)
                
                # Find EER
                fnr = 1 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                group_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
                
                demo_results[demo_name][str(value)] = {
                    'auc': group_auc,
                    'eer': group_eer,
                    'n_samples': np.sum(mask)
                }
        
        return demo_results
    
    def _evaluate_identification_by_demographics(self,
                                               similarity_matrix: np.ndarray,
                                               query_identities: np.ndarray,
                                               gallery_identities: np.ndarray,
                                               demographic_data: Dict,
                                               ranks: List[int]) -> Dict[str, Any]:
        """Evaluate identification performance by demographic groups"""
        demo_results = {}
        
        for demo_name, demo_values in demographic_data.items():
            demo_results[demo_name] = {}
            
            # Get unique demographic values
            unique_values = np.unique(demo_values)
            
            for value in unique_values:
                # Filter queries for this demographic group
                mask = demo_values == value
                if np.sum(mask) < 10:  # Skip groups with too few samples
                    continue
                
                group_query_ids = query_identities[mask]
                group_similarities = similarity_matrix[mask]
                
                # Compute rank accuracies for this group
                rank_accuracies = {}
                for rank in ranks:
                    correct = 0
                    for i, query_id in enumerate(group_query_ids):
                        top_k_indices = np.argsort(group_similarities[i])[::-1][:rank]
                        top_k_gallery_ids = gallery_identities[top_k_indices]
                        
                        if query_id in top_k_gallery_ids:
                            correct += 1
                    
                    rank_accuracies[f'Rank-{rank}'] = correct / len(group_query_ids)
                
                demo_results[demo_name][str(value)] = {
                    **rank_accuracies,
                    'n_samples': np.sum(mask)
                }
        
        return demo_results
    
    def evaluate_by_attributes(self,
                             similarities: np.ndarray,
                             labels: np.ndarray,
                             attributes: np.ndarray,
                             attribute_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate verification performance by attribute conditions
        
        Args:
            similarities: Similarity scores [N]
            labels: True labels [N]
            attributes: Attribute matrix [N, n_attributes]
            attribute_names: List of attribute names
            
        Returns:
            Dictionary with attribute-conditional metrics
        """
        results = {}
        
        for i, attr_name in enumerate(attribute_names):
            results[attr_name] = {}
            
            # Evaluate for presence (1) and absence (0) of attribute
            for attr_value in [0, 1]:
                mask = attributes[:, i] == attr_value
                if np.sum(mask) < 10:  # Skip if too few samples
                    continue
                
                attr_similarities = similarities[mask]
                attr_labels = labels[mask]
                
                # Compute metrics
                fpr, tpr, _ = roc_curve(attr_labels, attr_similarities)
                attr_auc = auc(fpr, tpr)
                
                # Find EER
                fnr = 1 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                attr_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
                
                results[attr_name][f'value_{attr_value}'] = {
                    'auc': attr_auc,
                    'eer': attr_eer,
                    'n_samples': np.sum(mask)
                }
        
        return results
    
    def create_roc_plot(self, 
                       results: Dict[str, Any],
                       save_path: Optional[str] = None) -> plt.Figure:
        """Create ROC curve plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot main ROC curve
        overall = results['overall']
        ax.plot(overall['fpr'], overall['tpr'], 
                linewidth=2, label=f'Overall (AUC = {overall["auc"]:.3f})')
        
        # Plot demographic group curves if available
        if 'demographic' in results:
            colors = plt.cm.Set1(np.linspace(0, 1, 10))
            color_idx = 0
            
            for demo_name, demo_data in results['demographic'].items():
                for group_name, group_metrics in demo_data.items():
                    if 'fpr' in group_metrics:  # Skip if insufficient data
                        ax.plot(group_metrics['fpr'], group_metrics['tpr'],
                               color=colors[color_idx % len(colors)],
                               label=f'{demo_name}={group_name} (AUC={group_metrics["auc"]:.3f})')
                        color_idx += 1
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_summary(self, 
                                 verification_results: Dict[str, Any],
                                 identification_results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create summary table of performance metrics"""
        summary_data = []
        
        # Verification metrics
        overall = verification_results['overall']
        summary_data.append({
            'Task': 'Verification',
            'Group': 'Overall',
            'AUC': overall['auc'],
            'EER': overall['eer'],
            'TAR@FAR=0.1%': overall['tar_at_far'].get('TAR@FAR=0.001', np.nan),
            'TAR@FAR=1%': overall['tar_at_far'].get('TAR@FAR=0.01', np.nan),
            'TAR@FAR=10%': overall['tar_at_far'].get('TAR@FAR=0.1', np.nan),
            'N_Samples': 'All'
        })
        
        # Demographic breakdown for verification
        if 'demographic' in verification_results:
            for demo_name, demo_data in verification_results['demographic'].items():
                for group_name, group_metrics in demo_data.items():
                    summary_data.append({
                        'Task': 'Verification',
                        'Group': f'{demo_name}={group_name}',
                        'AUC': group_metrics['auc'],
                        'EER': group_metrics['eer'],
                        'TAR@FAR=0.1%': np.nan,
                        'TAR@FAR=1%': np.nan,
                        'TAR@FAR=10%': np.nan,
                        'N_Samples': group_metrics['n_samples']
                    })
        
        # Identification metrics if provided
        if identification_results:
            overall_id = identification_results['overall']
            summary_data.append({
                'Task': 'Identification',
                'Group': 'Overall',
                'AUC': np.nan,
                'EER': np.nan,
                'TAR@FAR=0.1%': np.nan,
                'TAR@FAR=1%': np.nan,
                'TAR@FAR=10%': np.nan,
                'Rank-1': overall_id.get('Rank-1', np.nan),
                'Rank-5': overall_id.get('Rank-5', np.nan),
                'N_Samples': 'All'
            })
        
        return pd.DataFrame(summary_data)
    
    def bootstrap_confidence_intervals(self,
                                     similarities: np.ndarray,
                                     labels: np.ndarray,
                                     metric: str = 'auc',
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for metrics
        
        Args:
            similarities: Similarity scores
            labels: True labels
            metric: Metric to compute ('auc', 'eer')
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_samples = len(similarities)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_similarities = similarities[indices]
            boot_labels = labels[indices]
            
            # Compute metric
            if metric == 'auc':
                fpr, tpr, _ = roc_curve(boot_labels, boot_similarities)
                boot_metric = auc(fpr, tpr)
            elif metric == 'eer':
                fpr, tpr, _ = roc_curve(boot_labels, boot_similarities)
                fnr = 1 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                boot_metric = (fpr[eer_idx] + fnr[eer_idx]) / 2
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            bootstrap_metrics.append(boot_metric)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        
        return lower_bound, upper_bound