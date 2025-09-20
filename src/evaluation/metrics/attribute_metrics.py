"""
Attribute Metrics Evaluator
==========================

Comprehensive evaluation of attribute prediction performance including:
- Binary classification metrics per attribute
- Multi-class classification for categorical attributes
- Confusion matrices and detailed analysis
- Attribute correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer
from typing import Dict, List, Tuple, Optional, Any
import torch
import warnings
warnings.filterwarnings('ignore')

class AttributeEvaluator:
    """Evaluates facial attribute prediction performance"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda',
                 attribute_names: Optional[List[str]] = None):
        """
        Initialize attribute evaluator
        
        Args:
            model: Face recognition model with attribute prediction
            device: Device to run evaluations on
            attribute_names: List of attribute names
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
        
        # Categorical attribute groups for age analysis
        self.categorical_attributes = {
            'age_group': {
                'attributes': ['Young'],
                'bins': ['Young', 'Middle-aged', 'Old'],
                'mapping': self._create_age_mapping
            },
            'hair_color': {
                'attributes': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
                'bins': ['Black', 'Blond', 'Brown', 'Gray', 'Other'],
                'mapping': self._create_hair_color_mapping
            }
        }
    
    def predict_attributes(self, 
                          images: torch.Tensor, 
                          batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict attributes for a batch of images
        
        Args:
            images: Input images [N, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (predictions, probabilities) [N, n_attributes]
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                outputs = self.model(batch)
                
                if 'attribute_logits' in outputs:
                    attr_logits = outputs['attribute_logits']
                    attr_probs = torch.sigmoid(attr_logits)
                    attr_preds = (attr_probs > 0.5).float()
                    
                    predictions.append(attr_preds.cpu().numpy())
                    probabilities.append(attr_probs.cpu().numpy())
                else:
                    raise ValueError("Model does not output attribute predictions")
        
        return np.vstack(predictions), np.vstack(probabilities)
    
    def evaluate_binary_attributes(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate binary attribute classification performance
        
        Args:
            y_true: True attribute labels [N, n_attributes]
            y_pred: Predicted attribute labels [N, n_attributes]
            y_prob: Predicted probabilities [N, n_attributes]
            
        Returns:
            Dictionary with per-attribute metrics
        """
        results = {}
        n_attributes = y_true.shape[1]
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
            'precision_macro': precision_score(y_true.flatten(), y_pred.flatten(), average='macro'),
            'recall_macro': recall_score(y_true.flatten(), y_pred.flatten(), average='macro'),
            'f1_macro': f1_score(y_true.flatten(), y_pred.flatten(), average='macro')
        }
        
        if y_prob is not None:
            overall_metrics['auc_macro'] = roc_auc_score(y_true.flatten(), y_prob.flatten())
        
        results['overall'] = overall_metrics
        
        # Per-attribute metrics
        per_attribute = {}
        for i, attr_name in enumerate(self.attribute_names[:n_attributes]):
            attr_true = y_true[:, i]
            attr_pred = y_pred[:, i]
            
            # Skip if all labels are the same (causes issues with some metrics)
            if len(np.unique(attr_true)) == 1:
                continue
            
            attr_metrics = {
                'accuracy': accuracy_score(attr_true, attr_pred),
                'precision': precision_score(attr_true, attr_pred, average='binary', zero_division=0),
                'recall': recall_score(attr_true, attr_pred, average='binary', zero_division=0),
                'f1': f1_score(attr_true, attr_pred, average='binary', zero_division=0),
                'support_positive': np.sum(attr_true == 1),
                'support_negative': np.sum(attr_true == 0),
                'prevalence': np.mean(attr_true)
            }
            
            # Add AUC if probabilities provided
            if y_prob is not None:
                attr_prob = y_prob[:, i]
                try:
                    attr_metrics['auc'] = roc_auc_score(attr_true, attr_prob)
                    attr_metrics['average_precision'] = average_precision_score(attr_true, attr_prob)
                except ValueError:
                    # Handle case where only one class present
                    attr_metrics['auc'] = np.nan
                    attr_metrics['average_precision'] = np.nan
            
            per_attribute[attr_name] = attr_metrics
        
        results['per_attribute'] = per_attribute
        
        # Attribute correlation analysis
        results['correlations'] = self._analyze_attribute_correlations(y_true, y_pred)
        
        return results
    
    def evaluate_categorical_attributes(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      attribute_group: str) -> Dict[str, Any]:
        """
        Evaluate categorical attribute performance (e.g., age groups)
        
        Args:
            y_true: True attribute labels [N, n_attributes]
            y_pred: Predicted attribute labels [N, n_attributes]
            attribute_group: Name of categorical attribute group
            
        Returns:
            Dictionary with categorical metrics
        """
        if attribute_group not in self.categorical_attributes:
            raise ValueError(f"Unknown categorical attribute group: {attribute_group}")
        
        cat_info = self.categorical_attributes[attribute_group]
        
        # Convert binary attributes to categorical labels
        true_categorical = cat_info['mapping'](y_true)
        pred_categorical = cat_info['mapping'](y_pred)
        
        # Compute metrics
        accuracy = accuracy_score(true_categorical, pred_categorical)
        
        # Confusion matrix
        labels = cat_info['bins']
        cm = confusion_matrix(true_categorical, pred_categorical, labels=labels)
        
        # Classification report
        report = classification_report(
            true_categorical, pred_categorical, 
            labels=labels, output_dict=True, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'labels': labels
        }
    
    def _create_age_mapping(self, attributes: np.ndarray) -> np.ndarray:
        """Create age group mapping from Young attribute"""
        young_idx = self.attribute_names.index('Young')
        young = attributes[:, young_idx]
        
        # Simple mapping: Young=1 -> 'Young', Young=0 -> 'Old'
        # In practice, would use more sophisticated age estimation
        age_groups = np.where(young == 1, 'Young', 'Old')
        
        return age_groups
    
    def _create_hair_color_mapping(self, attributes: np.ndarray) -> np.ndarray:
        """Create hair color mapping from hair color attributes"""
        hair_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        hair_indices = [self.attribute_names.index(attr) for attr in hair_attrs]
        
        hair_colors = []
        for i in range(len(attributes)):
            # Find which hair color attributes are present
            hair_present = attributes[i, hair_indices]
            
            if hair_present[0]:  # Black_Hair
                hair_colors.append('Black')
            elif hair_present[1]:  # Blond_Hair
                hair_colors.append('Blond')
            elif hair_present[2]:  # Brown_Hair
                hair_colors.append('Brown')
            elif hair_present[3]:  # Gray_Hair
                hair_colors.append('Gray')
            else:
                hair_colors.append('Other')
        
        return np.array(hair_colors)
    
    def _analyze_attribute_correlations(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations between attributes"""
        
        # True attribute correlations
        true_corr = np.corrcoef(y_true.T)
        
        # Predicted attribute correlations  
        pred_corr = np.corrcoef(y_pred.T)
        
        # Correlation between true and predicted correlations
        correlation_agreement = np.corrcoef(
            true_corr.flatten(), pred_corr.flatten()
        )[0, 1]
        
        return {
            'true_correlations': true_corr,
            'predicted_correlations': pred_corr,
            'correlation_agreement': correlation_agreement
        }
    
    def create_attribute_performance_plot(self,
                                        results: Dict[str, Any],
                                        metric: str = 'f1',
                                        save_path: Optional[str] = None) -> plt.Figure:
        """Create bar plot of per-attribute performance"""
        per_attr = results['per_attribute']
        
        # Extract metric values and names
        attr_names = []
        metric_values = []
        
        for attr_name, metrics in per_attr.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                attr_names.append(attr_name.replace('_', ' '))
                metric_values.append(metrics[metric])
        
        # Sort by performance
        sorted_indices = np.argsort(metric_values)[::-1]
        attr_names = [attr_names[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 8))
        bars = ax.bar(range(len(attr_names)), metric_values)
        
        # Color bars by performance level
        colors = plt.cm.RdYlGn([v for v in metric_values])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Attributes')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'Per-Attribute {metric.upper()} Performance')
        ax.set_xticks(range(len(attr_names)))
        ax.set_xticklabels(attr_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add performance threshold line
        threshold = 0.7 if metric in ['f1', 'precision', 'recall'] else 0.8
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Target ({threshold})')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_confusion_matrix_plot(self,
                                   results: Dict[str, Any],
                                   attribute_group: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create confusion matrix heatmap for categorical attributes"""
        
        cm = results['confusion_matrix']
        labels = results['labels']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title(f'{attribute_group.title()} - Raw Counts')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title(f'{attribute_group.title()} - Normalized')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_correlation_heatmap(self,
                                 results: Dict[str, Any],
                                 correlation_type: str = 'true',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create attribute correlation heatmap"""
        
        correlations = results['correlations']
        
        if correlation_type == 'true':
            corr_matrix = correlations['true_correlations']
            title = 'True Attribute Correlations'
        elif correlation_type == 'predicted':
            corr_matrix = correlations['predicted_correlations']
            title = 'Predicted Attribute Correlations'
        else:
            raise ValueError("correlation_type must be 'true' or 'predicted'")
        
        # Use subset of attributes for readability
        n_attrs = min(20, len(self.attribute_names))
        subset_names = [name.replace('_', ' ') for name in self.attribute_names[:n_attrs]]
        subset_corr = corr_matrix[:n_attrs, :n_attrs]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(subset_corr, dtype=bool))
        sns.heatmap(subset_corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True,
                   xticklabels=subset_names, yticklabels=subset_names,
                   ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_summary_table(self,
                                        results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table of attribute performance"""
        
        per_attr = results['per_attribute']
        overall = results['overall']
        
        # Create summary data
        summary_data = []
        
        # Add overall metrics
        summary_data.append({
            'Attribute': 'OVERALL',
            'Accuracy': overall['accuracy'],
            'Precision': overall['precision_macro'],
            'Recall': overall['recall_macro'],
            'F1': overall['f1_macro'],
            'AUC': overall.get('auc_macro', np.nan),
            'Support+': 'All',
            'Support-': 'All',
            'Prevalence': np.nan
        })
        
        # Add per-attribute metrics
        for attr_name, metrics in per_attr.items():
            summary_data.append({
                'Attribute': attr_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'AUC': metrics.get('auc', np.nan),
                'Support+': metrics['support_positive'],
                'Support-': metrics['support_negative'],
                'Prevalence': metrics['prevalence']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by F1 score (excluding overall)
        df_overall = df.iloc[:1]
        df_attrs = df.iloc[1:].sort_values('F1', ascending=False)
        df = pd.concat([df_overall, df_attrs], ignore_index=True)
        
        return df
    
    def identify_problematic_attributes(self,
                                      results: Dict[str, Any],
                                      threshold: float = 0.7) -> Dict[str, List[str]]:
        """Identify attributes with poor performance"""
        
        per_attr = results['per_attribute']
        
        problematic = {
            'low_f1': [],
            'low_precision': [],
            'low_recall': [],
            'imbalanced': []
        }
        
        for attr_name, metrics in per_attr.items():
            # Low F1 score
            if metrics['f1'] < threshold:
                problematic['low_f1'].append(attr_name)
            
            # Low precision
            if metrics['precision'] < threshold:
                problematic['low_precision'].append(attr_name)
            
            # Low recall
            if metrics['recall'] < threshold:
                problematic['low_recall'].append(attr_name)
            
            # Imbalanced classes (prevalence < 10% or > 90%)
            if metrics['prevalence'] < 0.1 or metrics['prevalence'] > 0.9:
                problematic['imbalanced'].append(attr_name)
        
        return problematic