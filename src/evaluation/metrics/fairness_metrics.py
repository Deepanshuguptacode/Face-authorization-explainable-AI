"""
Fairness Metrics Evaluator
==========================

Comprehensive fairness evaluation including:
- TPR/FPR gap analysis across demographic groups
- Statistical significance testing for bias detection
- Demographic performance disaggregation
- Equalized odds and equal opportunity metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Any
import torch
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FairnessEvaluator:
    """Evaluates fairness across demographic groups"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda'):
        """
        Initialize fairness evaluator
        
        Args:
            model: Face recognition model
            device: Device to run evaluations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_group_performance_gaps(self,
                                     verification_results: Dict[str, Any],
                                     demographic_data: pd.DataFrame,
                                     protected_attributes: List[str] = ['Male', 'Young'],
                                     threshold: float = 0.5,
                                     n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Compute performance gaps between demographic groups
        
        Args:
            verification_results: Results from verification evaluation
            demographic_data: DataFrame with demographic attributes
            protected_attributes: List of binary demographic attributes
            threshold: Decision threshold for binary classification
            n_bootstrap: Number of bootstrap samples for confidence intervals
            
        Returns:
            Dictionary with fairness gap analysis
        """
        results = {
            'protected_attributes': protected_attributes,
            'threshold': threshold,
            'group_performance': {},
            'pairwise_gaps': {},
            'statistical_tests': {}
        }
        
        similarities = verification_results['similarities']
        labels = verification_results['labels']
        
        print("Computing group-wise performance metrics...")
        
        # For each protected attribute
        for attr in protected_attributes:
            if attr not in demographic_data.columns:
                print(f"Warning: {attr} not found in demographic data")
                continue
            
            results['group_performance'][attr] = {}
            results['pairwise_gaps'][attr] = {}
            results['statistical_tests'][attr] = {}
            
            # Get group indicators
            group_1_mask = demographic_data[attr] == 1
            group_0_mask = demographic_data[attr] == 0
            
            group_1_indices = group_1_mask[group_1_mask].index.tolist()
            group_0_indices = group_0_mask[group_0_mask].index.tolist()
            
            # Extract similarities and labels for each group
            group_1_similarities = similarities[group_1_indices]
            group_1_labels = labels[group_1_indices]
            
            group_0_similarities = similarities[group_0_indices]
            group_0_labels = labels[group_0_indices]
            
            # Compute binary classification metrics for each group
            group_1_metrics = self._compute_binary_metrics(
                group_1_similarities, group_1_labels, threshold
            )
            group_0_metrics = self._compute_binary_metrics(
                group_0_similarities, group_0_labels, threshold
            )
            
            results['group_performance'][attr]['group_1'] = group_1_metrics
            results['group_performance'][attr]['group_0'] = group_0_metrics
            
            # Compute gaps
            tpr_gap = group_1_metrics['tpr'] - group_0_metrics['tpr']
            fpr_gap = group_1_metrics['fpr'] - group_0_metrics['fpr']
            accuracy_gap = group_1_metrics['accuracy'] - group_0_metrics['accuracy']
            
            results['pairwise_gaps'][attr] = {
                'tpr_gap': tpr_gap,
                'fpr_gap': fpr_gap,
                'accuracy_gap': accuracy_gap,
                'group_1_size': len(group_1_indices),
                'group_0_size': len(group_0_indices)
            }
            
            # Statistical significance testing
            print(f"Testing statistical significance for {attr}...")
            
            # Bootstrap confidence intervals for gaps
            bootstrap_tpr_gaps = []
            bootstrap_fpr_gaps = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement from each group
                g1_sample_idx = np.random.choice(len(group_1_similarities), 
                                               len(group_1_similarities), replace=True)
                g0_sample_idx = np.random.choice(len(group_0_similarities),
                                               len(group_0_similarities), replace=True)
                
                g1_sample_sim = group_1_similarities[g1_sample_idx]
                g1_sample_labels = group_1_labels[g1_sample_idx]
                g0_sample_sim = group_0_similarities[g0_sample_idx]
                g0_sample_labels = group_0_labels[g0_sample_idx]
                
                # Compute metrics
                g1_metrics = self._compute_binary_metrics(g1_sample_sim, g1_sample_labels, threshold)
                g0_metrics = self._compute_binary_metrics(g0_sample_sim, g0_sample_labels, threshold)
                
                bootstrap_tpr_gaps.append(g1_metrics['tpr'] - g0_metrics['tpr'])
                bootstrap_fpr_gaps.append(g1_metrics['fpr'] - g0_metrics['fpr'])
            
            # Confidence intervals (95%)
            tpr_gap_ci = (np.percentile(bootstrap_tpr_gaps, 2.5),
                         np.percentile(bootstrap_tpr_gaps, 97.5))
            fpr_gap_ci = (np.percentile(bootstrap_fpr_gaps, 2.5),
                         np.percentile(bootstrap_fpr_gaps, 97.5))
            
            # Permutation test for significance
            tpr_pvalue = self._permutation_test(
                group_1_similarities, group_1_labels,
                group_0_similarities, group_0_labels,
                threshold, metric='tpr'
            )
            
            fpr_pvalue = self._permutation_test(
                group_1_similarities, group_1_labels,
                group_0_similarities, group_0_labels,
                threshold, metric='fpr'
            )
            
            results['statistical_tests'][attr] = {
                'tpr_gap_ci': tpr_gap_ci,
                'fpr_gap_ci': fpr_gap_ci,
                'tpr_gap_pvalue': tpr_pvalue,
                'fpr_gap_pvalue': fpr_pvalue,
                'tpr_gap_significant': tpr_pvalue < 0.05,
                'fpr_gap_significant': fpr_pvalue < 0.05
            }
        
        return results
    
    def _compute_binary_metrics(self,
                               similarities: np.ndarray,
                               labels: np.ndarray,
                               threshold: float) -> Dict[str, float]:
        """Compute binary classification metrics"""
        predictions = (similarities >= threshold).astype(int)
        
        # Confusion matrix components
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tp = np.sum((predictions == 1) & (labels == 1))
        
        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def _permutation_test(self,
                         group_1_similarities: np.ndarray,
                         group_1_labels: np.ndarray,
                         group_0_similarities: np.ndarray,
                         group_0_labels: np.ndarray,
                         threshold: float,
                         metric: str = 'tpr',
                         n_permutations: int = 1000) -> float:
        """Permutation test for group difference significance"""
        
        # Observed difference
        g1_metrics = self._compute_binary_metrics(group_1_similarities, group_1_labels, threshold)
        g0_metrics = self._compute_binary_metrics(group_0_similarities, group_0_labels, threshold)
        observed_diff = g1_metrics[metric] - g0_metrics[metric]
        
        # Combine all data
        all_similarities = np.concatenate([group_1_similarities, group_0_similarities])
        all_labels = np.concatenate([group_1_labels, group_0_labels])
        
        n_group_1 = len(group_1_similarities)
        n_total = len(all_similarities)
        
        # Permutation test
        null_diffs = []
        for _ in range(n_permutations):
            # Randomly reassign group membership
            perm_indices = np.random.permutation(n_total)
            perm_g1_sim = all_similarities[perm_indices[:n_group_1]]
            perm_g1_labels = all_labels[perm_indices[:n_group_1]]
            perm_g0_sim = all_similarities[perm_indices[n_group_1:]]
            perm_g0_labels = all_labels[perm_indices[n_group_1:]]
            
            # Compute metrics
            perm_g1_metrics = self._compute_binary_metrics(perm_g1_sim, perm_g1_labels, threshold)
            perm_g0_metrics = self._compute_binary_metrics(perm_g0_sim, perm_g0_labels, threshold)
            
            null_diff = perm_g1_metrics[metric] - perm_g0_metrics[metric]
            null_diffs.append(null_diff)
        
        # Two-tailed p-value
        null_diffs = np.array(null_diffs)
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
        
        return p_value
    
    def compute_intersectional_fairness(self,
                                      verification_results: Dict[str, Any],
                                      demographic_data: pd.DataFrame,
                                      intersectional_groups: List[List[str]],
                                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze fairness across intersectional demographic groups
        
        Args:
            verification_results: Results from verification evaluation
            demographic_data: DataFrame with demographic attributes
            intersectional_groups: List of attribute combinations for intersectional analysis
            threshold: Decision threshold
            
        Returns:
            Dictionary with intersectional fairness analysis
        """
        results = {
            'intersectional_groups': intersectional_groups,
            'group_performance': {},
            'performance_matrix': None
        }
        
        similarities = verification_results['similarities']
        labels = verification_results['labels']
        
        print("Computing intersectional fairness metrics...")
        
        # Create group combinations
        group_performance_data = []
        
        for group_attrs in intersectional_groups:
            # Create group identifier
            group_name = ' & '.join([f"{attr}=1" for attr in group_attrs])
            
            # Find samples belonging to this intersectional group
            mask = np.ones(len(demographic_data), dtype=bool)
            for attr in group_attrs:
                if attr in demographic_data.columns:
                    mask &= (demographic_data[attr] == 1)
                else:
                    print(f"Warning: {attr} not found in demographic data")
                    mask &= False
            
            if np.sum(mask) == 0:
                print(f"Warning: No samples found for group {group_name}")
                continue
            
            group_indices = demographic_data[mask].index.tolist()
            group_similarities = similarities[group_indices]
            group_labels = labels[group_indices]
            
            # Compute metrics
            metrics = self._compute_binary_metrics(group_similarities, group_labels, threshold)
            metrics['group_size'] = len(group_indices)
            metrics['group_name'] = group_name
            
            results['group_performance'][group_name] = metrics
            group_performance_data.append(metrics)
        
        # Create performance comparison matrix
        if group_performance_data:
            df_data = []
            for group_data in group_performance_data:
                df_data.append({
                    'Group': group_data['group_name'],
                    'Size': group_data['group_size'],
                    'TPR': group_data['tpr'],
                    'FPR': group_data['fpr'],
                    'Accuracy': group_data['accuracy'],
                    'F1': group_data['f1']
                })
            
            results['performance_matrix'] = pd.DataFrame(df_data)
        
        return results
    
    def compute_equalized_odds_metrics(self,
                                     verification_results: Dict[str, Any],
                                     demographic_data: pd.DataFrame,
                                     protected_attribute: str = 'Male',
                                     threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute equalized odds and equal opportunity metrics
        
        Args:
            verification_results: Results from verification evaluation
            demographic_data: DataFrame with demographic attributes
            protected_attribute: Protected attribute for fairness analysis
            threshold: Decision threshold
            
        Returns:
            Dictionary with equalized odds analysis
        """
        similarities = verification_results['similarities']
        labels = verification_results['labels']
        
        if protected_attribute not in demographic_data.columns:
            raise ValueError(f"{protected_attribute} not found in demographic data")
        
        # Group indicators
        group_1_mask = demographic_data[protected_attribute] == 1
        group_0_mask = demographic_data[protected_attribute] == 0
        
        group_1_indices = group_1_mask[group_1_mask].index.tolist()
        group_0_indices = group_0_mask[group_0_mask].index.tolist()
        
        # Extract data for each group
        g1_similarities = similarities[group_1_indices]
        g1_labels = labels[group_1_indices]
        g0_similarities = similarities[group_0_indices]
        g0_labels = labels[group_0_indices]
        
        # Compute metrics
        g1_metrics = self._compute_binary_metrics(g1_similarities, g1_labels, threshold)
        g0_metrics = self._compute_binary_metrics(g0_similarities, g0_labels, threshold)
        
        # Equalized odds: TPR and FPR should be equal across groups
        tpr_diff = abs(g1_metrics['tpr'] - g0_metrics['tpr'])
        fpr_diff = abs(g1_metrics['fpr'] - g0_metrics['fpr'])
        equalized_odds_violation = max(tpr_diff, fpr_diff)
        
        # Equal opportunity: TPR should be equal across groups
        equal_opportunity_violation = tpr_diff
        
        # Demographic parity: Overall positive rate should be equal
        g1_positive_rate = np.mean(g1_similarities >= threshold)
        g0_positive_rate = np.mean(g0_similarities >= threshold)
        demographic_parity_violation = abs(g1_positive_rate - g0_positive_rate)
        
        results = {
            'protected_attribute': protected_attribute,
            'threshold': threshold,
            'group_1_metrics': g1_metrics,
            'group_0_metrics': g0_metrics,
            'fairness_violations': {
                'equalized_odds': equalized_odds_violation,
                'equal_opportunity': equal_opportunity_violation,
                'demographic_parity': demographic_parity_violation
            },
            'detailed_gaps': {
                'tpr_gap': g1_metrics['tpr'] - g0_metrics['tpr'],
                'fpr_gap': g1_metrics['fpr'] - g0_metrics['fpr'],
                'accuracy_gap': g1_metrics['accuracy'] - g0_metrics['accuracy'],
                'positive_rate_gap': g1_positive_rate - g0_positive_rate
            }
        }
        
        return results
    
    def bias_audit_report(self,
                         verification_results: Dict[str, Any],
                         attribute_results: Dict[str, Any],
                         demographic_data: pd.DataFrame,
                         protected_attributes: List[str] = ['Male', 'Young'],
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive bias audit report
        
        Args:
            verification_results: Face verification results
            attribute_results: Attribute prediction results
            demographic_data: Demographic attribute data
            protected_attributes: List of protected attributes to analyze
            save_path: Path to save detailed report
            
        Returns:
            Dictionary with comprehensive bias audit
        """
        audit_results = {
            'executive_summary': {},
            'verification_bias': {},
            'attribute_bias': {},
            'recommendations': []
        }
        
        print("Generating comprehensive bias audit report...")
        
        # Verification bias analysis
        verification_gaps = self.compute_group_performance_gaps(
            verification_results, demographic_data, protected_attributes
        )
        audit_results['verification_bias'] = verification_gaps
        
        # Attribute prediction bias analysis
        attribute_bias = {}
        for attr in protected_attributes:
            if attr in demographic_data.columns:
                # Analyze bias in predicting this attribute
                if attr in attribute_results['individual_results']:
                    attr_predictions = attribute_results['individual_results'][attr]['predictions']
                    attr_true_labels = attribute_results['individual_results'][attr]['true_labels']
                    
                    # Group-wise attribute prediction accuracy
                    group_1_mask = demographic_data[attr] == 1
                    group_0_mask = demographic_data[attr] == 0
                    
                    g1_accuracy = np.mean(
                        attr_predictions[group_1_mask] == attr_true_labels[group_1_mask]
                    )
                    g0_accuracy = np.mean(
                        attr_predictions[group_0_mask] == attr_true_labels[group_0_mask]
                    )
                    
                    attribute_bias[attr] = {
                        'group_1_accuracy': g1_accuracy,
                        'group_0_accuracy': g0_accuracy,
                        'accuracy_gap': g1_accuracy - g0_accuracy
                    }
        
        audit_results['attribute_bias'] = attribute_bias
        
        # Executive summary
        max_tpr_gap = max([
            abs(audit_results['verification_bias']['pairwise_gaps'][attr]['tpr_gap'])
            for attr in protected_attributes
            if attr in audit_results['verification_bias']['pairwise_gaps']
        ])
        
        max_fpr_gap = max([
            abs(audit_results['verification_bias']['pairwise_gaps'][attr]['fpr_gap'])
            for attr in protected_attributes
            if attr in audit_results['verification_bias']['pairwise_gaps']
        ])
        
        audit_results['executive_summary'] = {
            'max_tpr_gap': max_tpr_gap,
            'max_fpr_gap': max_fpr_gap,
            'bias_severity': self._classify_bias_severity(max_tpr_gap, max_fpr_gap),
            'protected_attributes_analyzed': protected_attributes,
            'significant_biases': self._identify_significant_biases(verification_gaps)
        }
        
        # Recommendations
        audit_results['recommendations'] = self._generate_bias_recommendations(
            max_tpr_gap, max_fpr_gap, audit_results['verification_bias']
        )
        
        # Save detailed report if requested
        if save_path:
            self._save_bias_audit_report(audit_results, save_path)
        
        return audit_results
    
    def _classify_bias_severity(self, max_tpr_gap: float, max_fpr_gap: float) -> str:
        """Classify bias severity based on performance gaps"""
        max_gap = max(abs(max_tpr_gap), abs(max_fpr_gap))
        
        if max_gap < 0.05:
            return "Low"
        elif max_gap < 0.1:
            return "Moderate"
        elif max_gap < 0.2:
            return "High"
        else:
            return "Critical"
    
    def _identify_significant_biases(self, verification_gaps: Dict[str, Any]) -> List[str]:
        """Identify statistically significant biases"""
        significant_biases = []
        
        for attr, tests in verification_gaps['statistical_tests'].items():
            if tests['tpr_gap_significant']:
                significant_biases.append(f"TPR gap for {attr}")
            if tests['fpr_gap_significant']:
                significant_biases.append(f"FPR gap for {attr}")
        
        return significant_biases
    
    def _generate_bias_recommendations(self,
                                     max_tpr_gap: float,
                                     max_fpr_gap: float,
                                     verification_gaps: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on bias analysis"""
        recommendations = []
        
        if max(abs(max_tpr_gap), abs(max_fpr_gap)) > 0.1:
            recommendations.append(
                "High bias detected. Consider data augmentation or algorithmic bias mitigation."
            )
        
        if abs(max_tpr_gap) > 0.05:
            recommendations.append(
                "Significant TPR gap indicates unequal benefit across groups. "
                "Review training data representation."
            )
        
        if abs(max_fpr_gap) > 0.05:
            recommendations.append(
                "Significant FPR gap indicates unequal harm across groups. "
                "Consider threshold calibration or fairness-aware training."
            )
        
        for attr, gaps in verification_gaps['pairwise_gaps'].items():
            if gaps['group_1_size'] < gaps['group_0_size'] * 0.1:
                recommendations.append(
                    f"Severely imbalanced groups for {attr}. "
                    "Collect more data for underrepresented group."
                )
        
        return recommendations
    
    def _save_bias_audit_report(self, audit_results: Dict[str, Any], save_path: str):
        """Save detailed bias audit report"""
        import json
        
        # Convert numpy types to JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep conversion
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = deep_convert(audit_results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Bias audit report saved to {save_path}")
    
    def create_fairness_visualization(self,
                                    fairness_results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive fairness visualization"""
        
        protected_attributes = fairness_results['protected_attributes']
        n_attrs = len(protected_attributes)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # TPR gaps
        tpr_gaps = [fairness_results['pairwise_gaps'][attr]['tpr_gap'] 
                   for attr in protected_attributes]
        
        axes[0].bar(protected_attributes, tpr_gaps, color='skyblue', edgecolor='black')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('True Positive Rate (TPR) Gaps')
        axes[0].set_ylabel('TPR Difference (Group 1 - Group 0)')
        axes[0].grid(True, alpha=0.3)
        
        # FPR gaps
        fpr_gaps = [fairness_results['pairwise_gaps'][attr]['fpr_gap']
                   for attr in protected_attributes]
        
        axes[1].bar(protected_attributes, fpr_gaps, color='lightcoral', edgecolor='black')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].set_title('False Positive Rate (FPR) Gaps')
        axes[1].set_ylabel('FPR Difference (Group 1 - Group 0)')
        axes[1].grid(True, alpha=0.3)
        
        # Performance comparison heatmap
        if len(protected_attributes) >= 2:
            # Create performance matrix
            perf_matrix = np.zeros((len(protected_attributes), 4))
            metrics = ['TPR', 'FPR', 'Accuracy', 'F1']
            
            for i, attr in enumerate(protected_attributes):
                g1_perf = fairness_results['group_performance'][attr]['group_1']
                perf_matrix[i, 0] = g1_perf['tpr']
                perf_matrix[i, 1] = g1_perf['fpr'] 
                perf_matrix[i, 2] = g1_perf['accuracy']
                perf_matrix[i, 3] = g1_perf['f1']
            
            im = axes[2].imshow(perf_matrix, cmap='RdYlBu_r', aspect='auto')
            axes[2].set_xticks(range(len(metrics)))
            axes[2].set_xticklabels(metrics)
            axes[2].set_yticks(range(len(protected_attributes)))
            axes[2].set_yticklabels([f"{attr}=1" for attr in protected_attributes])
            axes[2].set_title('Group Performance Heatmap')
            
            # Add text annotations
            for i in range(len(protected_attributes)):
                for j in range(len(metrics)):
                    text = axes[2].text(j, i, f'{perf_matrix[i, j]:.3f}',
                                      ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=axes[2])
        
        # Significance indicators
        significance_data = []
        for attr in protected_attributes:
            tests = fairness_results['statistical_tests'][attr]
            significance_data.append([
                1 if tests['tpr_gap_significant'] else 0,
                1 if tests['fpr_gap_significant'] else 0
            ])
        
        significance_matrix = np.array(significance_data)
        
        axes[3].imshow(significance_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(['TPR Gap', 'FPR Gap'])
        axes[3].set_yticks(range(len(protected_attributes)))
        axes[3].set_yticklabels(protected_attributes)
        axes[3].set_title('Statistical Significance (p < 0.05)')
        
        # Add text annotations
        for i in range(len(protected_attributes)):
            for j in range(2):
                is_sig = significance_matrix[i, j]
                text = "Sig" if is_sig else "n.s."
                color = "white" if is_sig else "black"
                axes[3].text(j, i, text, ha="center", va="center", color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig