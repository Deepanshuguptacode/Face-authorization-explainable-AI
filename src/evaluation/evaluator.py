"""
Evaluation Module for Face Recognition
=====================================

This module provides comprehensive evaluation metrics:
- Identity verification (ROC, AUC, EER)
- Attribute prediction accuracy
- Face recognition accuracy (rank-1, rank-5)
- Explainability analysis
- Bias detection across demographics
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class FaceRecognitionEvaluator:
    """
    Comprehensive evaluator for face recognition models
    """
    
    def __init__(self, 
                 model,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 attribute_names: Optional[List[str]] = None):
        """
        Args:
            model: Trained face recognition model
            device: Evaluation device
            attribute_names: Names of facial attributes
        """
        self.model = model.to(device)
        self.device = device
        self.attribute_names = attribute_names or [f'attr_{i}' for i in range(40)]
        
        # Results storage
        self.results = {}
    
    def extract_features(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract features, identity labels, attribute labels, and image IDs
        
        Args:
            data_loader: Data loader
        
        Returns:
            Tuple of (features, identities, attributes, image_ids)
        """
        self.model.eval()
        
        all_features = []
        all_identities = []
        all_attributes = []
        all_image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                images = batch['image'].to(self.device)
                
                # Extract embeddings
                features = self.model.extract_embeddings(images)
                all_features.append(features.cpu().numpy())
                
                # Get labels
                if 'identity' in batch:
                    all_identities.append(batch['identity'].numpy())
                
                if 'attributes' in batch:
                    all_attributes.append(batch['attributes'].numpy())
                
                all_image_ids.extend(batch['image_id'])
        
        # Concatenate results
        features = np.concatenate(all_features, axis=0)
        identities = np.concatenate(all_identities, axis=0) if all_identities else None
        attributes = np.concatenate(all_attributes, axis=0) if all_attributes else None
        
        return features, identities, attributes, all_image_ids
    
    def evaluate_verification(self, 
                            features1: np.ndarray,
                            features2: np.ndarray,
                            labels: np.ndarray,
                            plot_roc: bool = True) -> Dict[str, float]:
        """
        Evaluate face verification performance
        
        Args:
            features1, features2: Face embeddings for pairs
            labels: Binary labels (1 for same person, 0 for different)
            plot_roc: Whether to plot ROC curve
        
        Returns:
            Verification metrics
        """
        # Compute similarities
        similarities = F.cosine_similarity(
            torch.tensor(features1), 
            torch.tensor(features2), 
            dim=1
        ).numpy()
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        # Accuracy at different thresholds
        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_idx]
        best_accuracy = np.max(tpr - fpr + 1) / 2
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(labels, similarities)
        pr_auc = average_precision_score(labels, similarities)
        
        metrics = {
            'roc_auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'pr_auc': pr_auc
        }
        
        # Plot ROC curve
        if plot_roc:
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=8, label=f'EER = {eer:.3f}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same Person', density=True)
            plt.hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different Person', density=True)
            plt.axvline(eer_threshold, color='red', linestyle='--', label=f'EER Threshold = {eer_threshold:.3f}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Density')
            plt.title('Similarity Distribution')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            # Accuracy vs threshold
            accuracies = []
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                acc = accuracy_score(labels, predictions)
                accuracies.append(acc)
            
            plt.plot(thresholds, accuracies)
            plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.3f}')
            plt.xlabel('Threshold')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Threshold')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return metrics
    
    def evaluate_identification(self,
                              query_features: np.ndarray,
                              gallery_features: np.ndarray,
                              query_labels: np.ndarray,
                              gallery_labels: np.ndarray,
                              ranks: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Evaluate face identification performance
        
        Args:
            query_features: Query face embeddings
            gallery_features: Gallery face embeddings
            query_labels: Query identity labels
            gallery_labels: Gallery identity labels
            ranks: Ranks to compute accuracy for
        
        Returns:
            Identification metrics
        """
        # Compute similarity matrix
        similarities = np.dot(query_features, gallery_features.T)
        
        # Get sorted indices for each query
        sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]
        
        # Compute rank-k accuracies
        metrics = {}
        
        for k in ranks:
            correct = 0
            for i, query_label in enumerate(query_labels):
                # Get top-k gallery predictions
                top_k_indices = sorted_indices[i, :k]
                top_k_labels = gallery_labels[top_k_indices]
                
                # Check if correct identity is in top-k
                if query_label in top_k_labels:
                    correct += 1
            
            accuracy = correct / len(query_labels)
            metrics[f'rank_{k}_accuracy'] = accuracy
        
        return metrics
    
    def evaluate_attributes(self,
                          data_loader,
                          plot_confusion: bool = True) -> Dict[str, Any]:
        """
        Evaluate attribute prediction performance
        
        Args:
            data_loader: Data loader with attribute labels
            plot_confusion: Whether to plot confusion matrices
        
        Returns:
            Attribute evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating attributes"):
                images = batch['image'].to(self.device)
                labels = batch['attributes'].to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = torch.sign(outputs['attribute_logits'])
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Convert to binary {0, 1}
        predictions_binary = (predictions + 1) / 2
        labels_binary = (labels + 1) / 2
        
        # Compute metrics for each attribute
        attribute_metrics = {}
        
        for i, attr_name in enumerate(self.attribute_names):
            attr_predictions = predictions_binary[:, i]
            attr_labels = labels_binary[:, i]
            
            # Basic metrics
            accuracy = accuracy_score(attr_labels, attr_predictions)
            
            # Handle case where all labels are the same class
            if len(np.unique(attr_labels)) > 1:
                precision, recall, _ = precision_recall_curve(attr_labels, attr_predictions)
                pr_auc = average_precision_score(attr_labels, attr_predictions)
                
                # ROC if possible
                try:
                    fpr, tpr, _ = roc_curve(attr_labels, attr_predictions)
                    roc_auc = auc(fpr, tpr)
                except:
                    roc_auc = np.nan
            else:
                pr_auc = np.nan
                roc_auc = np.nan
            
            attribute_metrics[attr_name] = {
                'accuracy': accuracy,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'positive_ratio': attr_labels.mean()
            }
        
        # Overall metrics
        overall_accuracy = accuracy_score(labels_binary.flatten(), predictions_binary.flatten())
        
        # Plot attribute performance
        if plot_confusion:
            self._plot_attribute_performance(attribute_metrics)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'attribute_metrics': attribute_metrics,
            'predictions': predictions,
            'labels': labels
        }
        
        return results
    
    def evaluate_bias(self,
                     data_loader,
                     demographic_attributes: List[str] = ['Male', 'Young']) -> Dict[str, Any]:
        """
        Evaluate model bias across demographic groups
        
        Args:
            data_loader: Data loader
            demographic_attributes: Attributes to analyze bias for
        
        Returns:
            Bias analysis results
        """
        # Get features and predictions
        features, identities, attributes, image_ids = self.extract_features(data_loader)
        
        # Get attribute predictions
        self.model.eval()
        all_attr_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting attribute predictions"):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                predictions = torch.sign(outputs['attribute_logits'])
                all_attr_predictions.append(predictions.cpu().numpy())
        
        attr_predictions = np.concatenate(all_attr_predictions, axis=0)
        
        # Convert to binary
        attr_labels_binary = (attributes + 1) / 2
        attr_pred_binary = (attr_predictions + 1) / 2
        
        # Analyze bias for demographic attributes
        bias_results = {}
        
        for demo_attr in demographic_attributes:
            if demo_attr in self.attribute_names:
                attr_idx = self.attribute_names.index(demo_attr)
                demo_labels = attr_labels_binary[:, attr_idx]
                
                # Split into demographic groups
                group_0_mask = demo_labels == 0  # e.g., Female, Old
                group_1_mask = demo_labels == 1  # e.g., Male, Young
                
                # Compute metrics for each group
                group_0_accuracy = []
                group_1_accuracy = []
                
                for i, attr_name in enumerate(self.attribute_names):
                    if attr_name != demo_attr:
                        # Group 0 accuracy
                        if group_0_mask.sum() > 0:
                            acc_0 = accuracy_score(
                                attr_labels_binary[group_0_mask, i],
                                attr_pred_binary[group_0_mask, i]
                            )
                            group_0_accuracy.append(acc_0)
                        
                        # Group 1 accuracy
                        if group_1_mask.sum() > 0:
                            acc_1 = accuracy_score(
                                attr_labels_binary[group_1_mask, i],
                                attr_pred_binary[group_1_mask, i]
                            )
                            group_1_accuracy.append(acc_1)
                
                # Compute bias metrics
                if group_0_accuracy and group_1_accuracy:
                    avg_acc_0 = np.mean(group_0_accuracy)
                    avg_acc_1 = np.mean(group_1_accuracy)
                    accuracy_gap = abs(avg_acc_0 - avg_acc_1)
                    
                    bias_results[demo_attr] = {
                        'group_0_accuracy': avg_acc_0,
                        'group_1_accuracy': avg_acc_1,
                        'accuracy_gap': accuracy_gap,
                        'group_0_size': group_0_mask.sum(),
                        'group_1_size': group_1_mask.sum()
                    }
        
        return bias_results
    
    def _plot_attribute_performance(self, attribute_metrics: Dict[str, Dict[str, float]]):
        """Plot attribute prediction performance"""
        
        # Extract metrics
        attr_names = list(attribute_metrics.keys())
        accuracies = [attribute_metrics[attr]['accuracy'] for attr in attr_names]
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [attr_names[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]
        
        # Plot
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(sorted_names)), sorted_accuracies)
        plt.xlabel('Attributes')
        plt.ylabel('Accuracy')
        plt.title('Attribute Prediction Accuracy')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if sorted_accuracies[i] >= 0.9:
                bar.set_color('green')
            elif sorted_accuracies[i] >= 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Plot distribution of accuracies
        plt.subplot(2, 1, 2)
        plt.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Attributes')
        plt.title('Distribution of Attribute Accuracies')
        plt.axvline(np.mean(accuracies), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(accuracies):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_explanation(self,
                           image_tensor: torch.Tensor,
                           top_k: int = 5) -> Dict[str, Any]:
        """
        Generate explanation for a single image prediction
        
        Args:
            image_tensor: Input image tensor [1, 3, H, W]
            top_k: Number of top attributes to include in explanation
        
        Returns:
            Explanation dictionary
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions
            outputs = self.model(image_tensor.to(self.device))
            
            # Get embeddings
            embeddings = self.model.extract_embeddings(image_tensor.to(self.device))
            
            # Attribute predictions and confidences
            attr_logits = outputs['attribute_logits'].cpu().numpy()[0]
            attr_probs = torch.sigmoid(outputs['attribute_logits']).cpu().numpy()[0]
            attr_predictions = np.sign(attr_logits)
            
            # Get top positive and negative attributes
            positive_indices = np.argsort(attr_logits)[::-1][:top_k]
            negative_indices = np.argsort(attr_logits)[:top_k]
            
            explanation = {
                'embeddings': embeddings.cpu().numpy()[0],
                'attribute_predictions': attr_predictions,
                'attribute_confidences': attr_probs,
                'top_positive_attributes': [
                    {
                        'name': self.attribute_names[i],
                        'confidence': attr_probs[i],
                        'prediction': attr_predictions[i]
                    }
                    for i in positive_indices
                ],
                'top_negative_attributes': [
                    {
                        'name': self.attribute_names[i],
                        'confidence': 1 - attr_probs[i],
                        'prediction': attr_predictions[i]
                    }
                    for i in negative_indices
                ]
            }
        
        return explanation
    
    def compare_models(self,
                      model_results: Dict[str, Dict[str, Any]],
                      metrics_to_compare: List[str] = ['roc_auc', 'overall_accuracy']) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            model_results: Dictionary of model_name -> results
            metrics_to_compare: Metrics to include in comparison
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            for metric in metrics_to_compare:
                if metric in results:
                    row[metric] = results[metric]
                else:
                    row[metric] = np.nan
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


def evaluate_model(model,
                  test_loader,
                  attribute_names: Optional[List[str]] = None,
                  save_results: bool = True,
                  results_dir: str = 'results') -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        test_loader: Test data loader
        attribute_names: Attribute names
        save_results: Whether to save results
        results_dir: Directory to save results
    
    Returns:
        Complete evaluation results
    """
    evaluator = FaceRecognitionEvaluator(model, attribute_names=attribute_names)
    
    print("Starting comprehensive evaluation...")
    
    # Extract features
    features, identities, attributes, image_ids = evaluator.extract_features(test_loader)
    
    results = {}
    
    # Attribute evaluation
    if attributes is not None:
        print("Evaluating attributes...")
        attr_results = evaluator.evaluate_attributes(test_loader)
        results['attributes'] = attr_results
    
    # Bias evaluation
    if attributes is not None:
        print("Evaluating bias...")
        bias_results = evaluator.evaluate_bias(test_loader)
        results['bias'] = bias_results
    
    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save numerical results
        results_file = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            json_results[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, dict):
                            json_results[key][subkey] = {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in subvalue.items()
                            }
                        else:
                            json_results[key][subkey] = subvalue
            
            import json
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    print("Face Recognition Evaluator module loaded successfully!")
    print("Available evaluation functions:")
    print("- evaluate_verification: Face verification performance")
    print("- evaluate_identification: Face identification performance")
    print("- evaluate_attributes: Attribute prediction performance")
    print("- evaluate_bias: Bias analysis across demographics")
    print("- generate_explanation: Single image explanation")
    print("- compare_models: Multi-model comparison")