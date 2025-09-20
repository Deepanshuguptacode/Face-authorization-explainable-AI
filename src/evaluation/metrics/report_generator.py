"""
Comprehensive Evaluation Report Generator
========================================

Creates detailed evaluation reports with:
- Statistical analysis and performance tables
- Visualization plots and charts
- Error analysis and failure mode identification
- Recommendations and improvement suggestions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EvaluationReportGenerator:
    """Generates comprehensive evaluation reports"""
    
    def __init__(self, report_name: str = "Face_Recognition_Evaluation"):
        """
        Initialize report generator
        
        Args:
            report_name: Name for the evaluation report
        """
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_comprehensive_report(self,
                                    recognition_results: Dict[str, Any],
                                    attribute_results: Dict[str, Any],
                                    explanation_results: Dict[str, Any],
                                    fairness_results: Dict[str, Any],
                                    robustness_results: Dict[str, Any],
                                    model_info: Dict[str, Any],
                                    dataset_info: Dict[str, Any],
                                    save_dir: str = "evaluation_reports") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            recognition_results: Face recognition evaluation results
            attribute_results: Attribute prediction evaluation results
            explanation_results: Explainability evaluation results
            fairness_results: Fairness analysis results
            robustness_results: Robustness testing results
            model_info: Information about the evaluated model
            dataset_info: Information about the evaluation dataset
            save_dir: Directory to save the report
            
        Returns:
            Dictionary with report metadata and paths
        """
        print("Generating comprehensive evaluation report...")
        
        # Create report directory
        report_dir = Path(save_dir) / f"{self.report_name}_{self.timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_metadata = {
            'report_name': self.report_name,
            'timestamp': self.timestamp,
            'report_dir': str(report_dir),
            'sections': {}
        }
        
        # 1. Executive Summary
        print("Creating executive summary...")
        summary_path = self._create_executive_summary(
            recognition_results, attribute_results, explanation_results,
            fairness_results, robustness_results, report_dir
        )
        report_metadata['sections']['executive_summary'] = summary_path
        
        # 2. Recognition Performance Analysis
        print("Creating recognition analysis...")
        recognition_path = self._create_recognition_analysis(
            recognition_results, report_dir
        )
        report_metadata['sections']['recognition_analysis'] = recognition_path
        
        # 3. Attribute Performance Analysis
        print("Creating attribute analysis...")
        attribute_path = self._create_attribute_analysis(
            attribute_results, report_dir
        )
        report_metadata['sections']['attribute_analysis'] = attribute_path
        
        # 4. Explainability Evaluation
        print("Creating explainability analysis...")
        explanation_path = self._create_explanation_analysis(
            explanation_results, report_dir
        )
        report_metadata['sections']['explanation_analysis'] = explanation_path
        
        # 5. Fairness Assessment
        print("Creating fairness analysis...")
        fairness_path = self._create_fairness_analysis(
            fairness_results, report_dir
        )
        report_metadata['sections']['fairness_analysis'] = fairness_path
        
        # 6. Robustness Evaluation
        print("Creating robustness analysis...")
        robustness_path = self._create_robustness_analysis(
            robustness_results, report_dir
        )
        report_metadata['sections']['robustness_analysis'] = robustness_path
        
        # 7. Error Analysis
        print("Creating error analysis...")
        error_path = self._create_error_analysis(
            recognition_results, attribute_results, report_dir
        )
        report_metadata['sections']['error_analysis'] = error_path
        
        # 8. Recommendations
        print("Creating recommendations...")
        recommendations_path = self._create_recommendations(
            recognition_results, attribute_results, fairness_results,
            robustness_results, report_dir
        )
        report_metadata['sections']['recommendations'] = recommendations_path
        
        # 9. Technical Appendix
        print("Creating technical appendix...")
        appendix_path = self._create_technical_appendix(
            model_info, dataset_info, report_dir
        )
        report_metadata['sections']['technical_appendix'] = appendix_path
        
        # 10. Master Report (HTML)
        print("Creating master report...")
        master_report_path = self._create_master_report(
            report_metadata, recognition_results, attribute_results,
            explanation_results, fairness_results, robustness_results,
            model_info, dataset_info, report_dir
        )
        report_metadata['master_report'] = master_report_path
        
        # Save metadata
        metadata_path = report_dir / "report_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(report_metadata, f, indent=2)
        
        print(f"Report generated successfully at: {report_dir}")
        return report_metadata
    
    def _create_executive_summary(self,
                                recognition_results: Dict[str, Any],
                                attribute_results: Dict[str, Any],
                                explanation_results: Dict[str, Any],
                                fairness_results: Dict[str, Any],
                                robustness_results: Dict[str, Any],
                                report_dir: Path) -> str:
        """Create executive summary"""
        
        summary_path = report_dir / "executive_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("FACE RECOGNITION SYSTEM - EVALUATION EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {self.timestamp}\n")
            f.write(f"Report Name: {self.report_name}\n\n")
            
            # Key Performance Metrics
            f.write("KEY PERFORMANCE METRICS\n")
            f.write("-" * 25 + "\n")
            
            # Recognition Performance
            if 'verification_metrics' in recognition_results:
                auc = recognition_results['verification_metrics']['auc']
                eer = recognition_results['verification_metrics']['eer']
                f.write(f"• Face Verification AUC: {auc:.4f}\n")
                f.write(f"• Equal Error Rate (EER): {eer:.4f}\n")
            
            if 'identification_metrics' in recognition_results:
                rank1 = recognition_results['identification_metrics']['rank_1_accuracy']
                rank5 = recognition_results['identification_metrics']['rank_5_accuracy']
                f.write(f"• Rank-1 Identification: {rank1:.4f}\n")
                f.write(f"• Rank-5 Identification: {rank5:.4f}\n")
            
            # Attribute Performance
            if 'overall_metrics' in attribute_results:
                avg_accuracy = attribute_results['overall_metrics']['average_accuracy']
                avg_f1 = attribute_results['overall_metrics']['average_f1']
                f.write(f"• Average Attribute Accuracy: {avg_accuracy:.4f}\n")
                f.write(f"• Average Attribute F1-Score: {avg_f1:.4f}\n")
            
            f.write("\n")
            
            # Fairness Assessment
            f.write("FAIRNESS ASSESSMENT\n")
            f.write("-" * 18 + "\n")
            
            if 'executive_summary' in fairness_results:
                bias_severity = fairness_results['executive_summary']['bias_severity']
                max_tpr_gap = fairness_results['executive_summary']['max_tpr_gap']
                max_fpr_gap = fairness_results['executive_summary']['max_fpr_gap']
                
                f.write(f"• Bias Severity Level: {bias_severity}\n")
                f.write(f"• Maximum TPR Gap: {max_tpr_gap:.4f}\n")
                f.write(f"• Maximum FPR Gap: {max_fpr_gap:.4f}\n")
                
                if fairness_results['executive_summary']['significant_biases']:
                    f.write("• Significant Biases Detected:\n")
                    for bias in fairness_results['executive_summary']['significant_biases']:
                        f.write(f"  - {bias}\n")
                else:
                    f.write("• No statistically significant biases detected\n")
            
            f.write("\n")
            
            # Robustness Summary
            f.write("ROBUSTNESS SUMMARY\n")
            f.write("-" * 17 + "\n")
            
            if 'critical_occlusion_threshold' in robustness_results:
                for occlusion_type, threshold in robustness_results['critical_occlusion_threshold'].items():
                    if threshold:
                        f.write(f"• {occlusion_type.title()} Occlusion Threshold: {threshold:.2f}\n")
                    else:
                        f.write(f"• {occlusion_type.title()} Occlusion: Robust across all tested levels\n")
            
            if 'domain_gap' in robustness_results:
                domain_gap = robustness_results['domain_gap']['relative_gap']
                f.write(f"• Domain Shift Impact: {domain_gap:.2%} performance drop\n")
            
            f.write("\n")
            
            # Overall Assessment
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 17 + "\n")
            
            # Determine overall grade
            overall_grade = self._compute_overall_grade(
                recognition_results, attribute_results, fairness_results, robustness_results
            )
            
            f.write(f"• Overall System Grade: {overall_grade}\n")
            
            # Key strengths and concerns
            strengths, concerns = self._identify_strengths_and_concerns(
                recognition_results, attribute_results, fairness_results, robustness_results
            )
            
            if strengths:
                f.write("\n• Key Strengths:\n")
                for strength in strengths:
                    f.write(f"  - {strength}\n")
            
            if concerns:
                f.write("\n• Key Concerns:\n")
                for concern in concerns:
                    f.write(f"  - {concern}\n")
        
        return str(summary_path)
    
    def _compute_overall_grade(self,
                             recognition_results: Dict[str, Any],
                             attribute_results: Dict[str, Any],
                             fairness_results: Dict[str, Any],
                             robustness_results: Dict[str, Any]) -> str:
        """Compute overall system grade"""
        
        scores = []
        
        # Recognition score (30%)
        if 'verification_metrics' in recognition_results:
            auc = recognition_results['verification_metrics']['auc']
            recognition_score = min(auc * 100, 100)  # Convert to 0-100 scale
            scores.append(('recognition', recognition_score, 0.3))
        
        # Attribute score (20%)
        if 'overall_metrics' in attribute_results:
            avg_f1 = attribute_results['overall_metrics']['average_f1']
            attribute_score = min(avg_f1 * 100, 100)
            scores.append(('attribute', attribute_score, 0.2))
        
        # Fairness score (30%)
        if 'executive_summary' in fairness_results:
            bias_severity = fairness_results['executive_summary']['bias_severity']
            fairness_score = {'Low': 95, 'Moderate': 80, 'High': 60, 'Critical': 30}.get(bias_severity, 50)
            scores.append(('fairness', fairness_score, 0.3))
        
        # Robustness score (20%)
        robustness_score = 70  # Default moderate score
        if 'critical_occlusion_threshold' in robustness_results:
            # Higher threshold = more robust
            thresholds = [t for t in robustness_results['critical_occlusion_threshold'].values() if t]
            if thresholds:
                avg_threshold = np.mean(thresholds)
                robustness_score = min(50 + avg_threshold * 100, 95)
        scores.append(('robustness', robustness_score, 0.2))
        
        # Weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        # Convert to letter grade
        if weighted_score >= 90:
            return "A (Excellent)"
        elif weighted_score >= 80:
            return "B (Good)"
        elif weighted_score >= 70:
            return "C (Satisfactory)"
        elif weighted_score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    def _identify_strengths_and_concerns(self,
                                       recognition_results: Dict[str, Any],
                                       attribute_results: Dict[str, Any],
                                       fairness_results: Dict[str, Any],
                                       robustness_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify key strengths and concerns"""
        
        strengths = []
        concerns = []
        
        # Recognition analysis
        if 'verification_metrics' in recognition_results:
            auc = recognition_results['verification_metrics']['auc']
            if auc > 0.95:
                strengths.append("Excellent face verification performance")
            elif auc < 0.85:
                concerns.append("Suboptimal face verification accuracy")
        
        # Fairness analysis
        if 'executive_summary' in fairness_results:
            bias_severity = fairness_results['executive_summary']['bias_severity']
            if bias_severity == 'Low':
                strengths.append("Low demographic bias across groups")
            elif bias_severity in ['High', 'Critical']:
                concerns.append(f"{bias_severity} demographic bias detected")
        
        # Robustness analysis
        if 'critical_occlusion_threshold' in robustness_results:
            thresholds = [t for t in robustness_results['critical_occlusion_threshold'].values() if t]
            if thresholds and min(thresholds) > 0.3:
                strengths.append("Strong robustness to occlusion")
            elif thresholds and min(thresholds) < 0.1:
                concerns.append("Poor robustness to occlusion attacks")
        
        # Attribute analysis
        if 'problematic_attributes' in attribute_results:
            problematic = attribute_results['problematic_attributes']
            if len(problematic) > 5:
                concerns.append("Multiple poorly performing attributes")
            elif len(problematic) == 0:
                strengths.append("Consistent attribute prediction performance")
        
        return strengths, concerns
    
    def _create_recognition_analysis(self,
                                   recognition_results: Dict[str, Any],
                                   report_dir: Path) -> str:
        """Create detailed recognition performance analysis"""
        
        analysis_path = report_dir / "recognition_analysis.md"
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        with open(analysis_path, 'w') as f:
            f.write("# Face Recognition Performance Analysis\n\n")
            
            # Verification Metrics
            if 'verification_metrics' in recognition_results:
                f.write("## Verification Performance\n\n")
                
                metrics = recognition_results['verification_metrics']
                f.write("| Metric | Value | Confidence Interval |\n")
                f.write("|--------|-------|---------------------|\n")
                
                for metric_name, value in metrics.items():
                    if metric_name.endswith('_ci'):
                        continue
                    
                    ci_key = f"{metric_name}_ci"
                    if ci_key in metrics:
                        ci = metrics[ci_key]
                        f.write(f"| {metric_name.upper()} | {value:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n")
                    else:
                        f.write(f"| {metric_name.upper()} | {value:.4f} | - |\n")
                
                f.write("\n")
            
            # Identification Metrics
            if 'identification_metrics' in recognition_results:
                f.write("## Identification Performance\n\n")
                
                id_metrics = recognition_results['identification_metrics']
                f.write("| Rank | Accuracy | Confidence Interval |\n")
                f.write("|------|----------|---------------------|\n")
                
                for rank in [1, 5, 10]:
                    key = f"rank_{rank}_accuracy"
                    if key in id_metrics:
                        acc = id_metrics[key]
                        ci_key = f"{key}_ci"
                        if ci_key in id_metrics:
                            ci = id_metrics[ci_key]
                            f.write(f"| Rank-{rank} | {acc:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n")
                        else:
                            f.write(f"| Rank-{rank} | {acc:.4f} | - |\n")
                
                f.write("\n")
            
            # Demographic Analysis
            if 'demographic_analysis' in recognition_results:
                f.write("## Demographic Performance Breakdown\n\n")
                
                demo_analysis = recognition_results['demographic_analysis']
                for attribute, groups in demo_analysis.items():
                    f.write(f"### {attribute}\n\n")
                    f.write("| Group | AUC | EER | Samples |\n")
                    f.write("|-------|-----|-----|----------|\n")
                    
                    for group_name, group_metrics in groups.items():
                        auc = group_metrics.get('auc', 0.0)
                        eer = group_metrics.get('eer', 0.0)
                        n_samples = group_metrics.get('n_samples', 0)
                        f.write(f"| {group_name} | {auc:.4f} | {eer:.4f} | {n_samples} |\n")
                    
                    f.write("\n")
        
        return str(analysis_path)
    
    def _create_attribute_analysis(self,
                                 attribute_results: Dict[str, Any],
                                 report_dir: Path) -> str:
        """Create detailed attribute performance analysis"""
        
        analysis_path = report_dir / "attribute_analysis.md"
        
        with open(analysis_path, 'w') as f:
            f.write("# Facial Attribute Prediction Analysis\n\n")
            
            # Overall Performance
            if 'overall_metrics' in attribute_results:
                f.write("## Overall Performance\n\n")
                
                overall = attribute_results['overall_metrics']
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for metric_name, value in overall.items():
                    f.write(f"| {metric_name.replace('_', ' ').title()} | {value:.4f} |\n")
                
                f.write("\n")
            
            # Individual Attribute Performance
            if 'individual_results' in attribute_results:
                f.write("## Individual Attribute Performance\n\n")
                
                individual = attribute_results['individual_results']
                f.write("| Attribute | Accuracy | Precision | Recall | F1-Score |\n")
                f.write("|-----------|----------|-----------|--------|-----------|\n")
                
                for attr_name, metrics in individual.items():
                    if 'metrics' in metrics:
                        m = metrics['metrics']
                        f.write(f"| {attr_name} | {m.get('accuracy', 0):.4f} | "
                               f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                               f"{m.get('f1', 0):.4f} |\n")
                
                f.write("\n")
            
            # Problematic Attributes
            if 'problematic_attributes' in attribute_results:
                f.write("## Problematic Attributes (F1 < 0.7)\n\n")
                
                problematic = attribute_results['problematic_attributes']
                if problematic:
                    for attr in problematic:
                        f.write(f"- {attr}\n")
                else:
                    f.write("No attributes with F1-score below 0.7 threshold.\n")
                
                f.write("\n")
            
            # Correlation Analysis
            if 'correlation_analysis' in attribute_results:
                f.write("## Attribute Correlation Analysis\n\n")
                f.write("High correlations between attributes may indicate:\n")
                f.write("- Potential data leakage\n")
                f.write("- Inherent biological relationships\n")
                f.write("- Model bias or shortcuts\n\n")
                
                # Add details about strong correlations if available
                corr_data = attribute_results['correlation_analysis']
                if 'strong_correlations' in corr_data:
                    f.write("### Strong Correlations (|r| > 0.7)\n\n")
                    for correlation in corr_data['strong_correlations']:
                        f.write(f"- {correlation}\n")
                    f.write("\n")
        
        return str(analysis_path)
    
    def _create_explanation_analysis(self,
                                   explanation_results: Dict[str, Any],
                                   report_dir: Path) -> str:
        """Create explainability evaluation analysis"""
        
        analysis_path = report_dir / "explanation_analysis.md"
        
        with open(analysis_path, 'w') as f:
            f.write("# Explainability Evaluation Analysis\n\n")
            
            # Sanity Check Results
            if 'method' in explanation_results:
                f.write("## Sanity Check Results\n\n")
                f.write("Tests whether explanations depend on trained model vs. input statistics.\n\n")
                
                method = explanation_results['method']
                mean_sim = explanation_results['mean_similarity']
                mean_corr = explanation_results['mean_correlation']
                p_value = explanation_results['similarity_pvalue']
                
                f.write(f"**Method Tested:** {method}\n\n")
                f.write(f"- Mean similarity with randomized model: {mean_sim:.4f}\n")
                f.write(f"- Mean rank correlation: {mean_corr:.4f}\n")
                f.write(f"- Statistical significance (p-value): {p_value:.4f}\n\n")
                
                if mean_sim < 0.1 and p_value < 0.05:
                    f.write("✅ **PASS**: Explanations show low similarity with randomized models\n\n")
                else:
                    f.write("❌ **FAIL**: Explanations may not depend on trained model\n\n")
            
            # Deletion/Insertion Test
            if 'deletion_auc' in explanation_results:
                f.write("## Fidelity Test Results\n\n")
                f.write("Deletion/insertion test measures explanation fidelity.\n\n")
                
                del_auc = explanation_results['deletion_auc']
                ins_auc = explanation_results['insertion_auc']
                
                f.write(f"- Deletion AUC: {del_auc:.4f}\n")
                f.write(f"- Insertion AUC: {ins_auc:.4f}\n\n")
                
                if del_auc < ins_auc:
                    f.write("✅ **Good fidelity**: Deletion curve below insertion curve\n\n")
                else:
                    f.write("⚠️ **Poor fidelity**: Unusual deletion/insertion relationship\n\n")
            
            # Localization Test
            if 'mean_face_focus' in explanation_results:
                f.write("## Localization Analysis\n\n")
                
                face_focus = explanation_results['mean_face_focus']
                f.write(f"- Mean face region focus: {face_focus:.4f}\n\n")
                
                if face_focus > 0.6:
                    f.write("✅ **Good localization**: Explanations focus on face regions\n\n")
                else:
                    f.write("⚠️ **Poor localization**: Explanations may focus on background\n\n")
            
            # Human Evaluation Framework
            if 'n_images' in explanation_results:
                f.write("## Human Evaluation Setup\n\n")
                f.write(f"Human evaluation materials prepared for {explanation_results['n_images']} images.\n")
                f.write(f"Protocol saved at: {explanation_results.get('protocol_path', 'N/A')}\n\n")
        
        return str(analysis_path)
    
    def _create_fairness_analysis(self,
                                fairness_results: Dict[str, Any],
                                report_dir: Path) -> str:
        """Create fairness assessment analysis"""
        
        analysis_path = report_dir / "fairness_analysis.md"
        
        with open(analysis_path, 'w') as f:
            f.write("# Fairness Assessment Analysis\n\n")
            
            # Executive Summary
            if 'executive_summary' in fairness_results:
                f.write("## Executive Summary\n\n")
                
                summary = fairness_results['executive_summary']
                f.write(f"**Bias Severity:** {summary['bias_severity']}\n")
                f.write(f"**Maximum TPR Gap:** {summary['max_tpr_gap']:.4f}\n")
                f.write(f"**Maximum FPR Gap:** {summary['max_fpr_gap']:.4f}\n\n")
                
                if summary['significant_biases']:
                    f.write("**Significant Biases Detected:**\n")
                    for bias in summary['significant_biases']:
                        f.write(f"- {bias}\n")
                else:
                    f.write("**No statistically significant biases detected.**\n")
                
                f.write("\n")
            
            # Detailed Group Analysis
            if 'pairwise_gaps' in fairness_results:
                f.write("## Demographic Group Analysis\n\n")
                
                gaps = fairness_results['pairwise_gaps']
                f.write("| Attribute | TPR Gap | FPR Gap | Accuracy Gap | Group 1 Size | Group 0 Size |\n")
                f.write("|-----------|---------|---------|--------------|--------------|---------------|\n")
                
                for attr, gap_data in gaps.items():
                    f.write(f"| {attr} | {gap_data['tpr_gap']:.4f} | "
                           f"{gap_data['fpr_gap']:.4f} | {gap_data['accuracy_gap']:.4f} | "
                           f"{gap_data['group_1_size']} | {gap_data['group_0_size']} |\n")
                
                f.write("\n")
            
            # Statistical Significance
            if 'statistical_tests' in fairness_results:
                f.write("## Statistical Significance Tests\n\n")
                
                tests = fairness_results['statistical_tests']
                f.write("| Attribute | TPR Gap Significant | FPR Gap Significant | TPR p-value | FPR p-value |\n")
                f.write("|-----------|---------------------|---------------------|-------------|-------------|\n")
                
                for attr, test_data in tests.items():
                    tpr_sig = "Yes" if test_data['tpr_gap_significant'] else "No"
                    fpr_sig = "Yes" if test_data['fpr_gap_significant'] else "No"
                    f.write(f"| {attr} | {tpr_sig} | {fpr_sig} | "
                           f"{test_data['tpr_gap_pvalue']:.4f} | {test_data['fpr_gap_pvalue']:.4f} |\n")
                
                f.write("\n")
            
            # Recommendations
            if 'recommendations' in fairness_results:
                f.write("## Fairness Recommendations\n\n")
                
                for recommendation in fairness_results['recommendations']:
                    f.write(f"- {recommendation}\n")
                
                f.write("\n")
        
        return str(analysis_path)
    
    def _create_robustness_analysis(self,
                                  robustness_results: Dict[str, Any],
                                  report_dir: Path) -> str:
        """Create robustness evaluation analysis"""
        
        analysis_path = report_dir / "robustness_analysis.md"
        
        with open(analysis_path, 'w') as f:
            f.write("# Robustness Evaluation Analysis\n\n")
            
            # Occlusion Robustness
            if 'critical_occlusion_threshold' in robustness_results:
                f.write("## Occlusion Robustness\n\n")
                
                thresholds = robustness_results['critical_occlusion_threshold']
                f.write("Critical thresholds (20% performance drop):\n\n")
                
                for occlusion_type, threshold in thresholds.items():
                    if threshold:
                        f.write(f"- **{occlusion_type.title()}**: {threshold:.2f} occlusion ratio\n")
                    else:
                        f.write(f"- **{occlusion_type.title()}**: Robust across all tested levels\n")
                
                f.write("\n")
            
            # Pose Robustness
            if 'pose_robustness_curves' in robustness_results:
                f.write("## Pose Variation Robustness\n\n")
                
                curves = robustness_results['pose_robustness_curves']
                baseline = robustness_results.get('baseline_accuracy', 0.0)
                
                for rotation_type, accuracies in curves.items():
                    min_accuracy = min(accuracies)
                    max_drop = baseline - min_accuracy
                    f.write(f"- **{rotation_type.title()}**: Maximum performance drop = {max_drop:.4f}\n")
                
                f.write("\n")
            
            # Illumination Robustness
            if 'brightness_robustness' in robustness_results:
                f.write("## Illumination Robustness\n\n")
                
                brightness_acc = robustness_results['brightness_robustness']
                contrast_acc = robustness_results['contrast_robustness']
                baseline = robustness_results.get('baseline_accuracy', 0.0)
                
                min_brightness = min(brightness_acc)
                min_contrast = min(contrast_acc)
                
                f.write(f"- **Brightness variation**: Minimum accuracy = {min_brightness:.4f} "
                       f"(drop: {baseline - min_brightness:.4f})\n")
                f.write(f"- **Contrast variation**: Minimum accuracy = {min_contrast:.4f} "
                       f"(drop: {baseline - min_contrast:.4f})\n\n")
            
            # Domain Shift Analysis
            if 'domain_gap' in robustness_results:
                f.write("## Domain Shift Analysis\n\n")
                
                domain_gap = robustness_results['domain_gap']
                f.write(f"- **Absolute performance gap**: {domain_gap['absolute_gap']:.4f}\n")
                f.write(f"- **Relative performance drop**: {domain_gap['relative_gap']:.2%}\n\n")
            
            # Attribute Stability
            if 'attribute_stability' in robustness_results:
                f.write("## Attribute Perturbation Stability\n\n")
                
                stability = robustness_results['attribute_stability']
                f.write("| Attribute | Stability Score |\n")
                f.write("|-----------|------------------|\n")
                
                for attr, score in stability.items():
                    f.write(f"| {attr} | {score:.4f} |\n")
                
                f.write("\n")
        
        return str(analysis_path)
    
    def _create_error_analysis(self,
                             recognition_results: Dict[str, Any],
                             attribute_results: Dict[str, Any],
                             report_dir: Path) -> str:
        """Create detailed error analysis"""
        
        analysis_path = report_dir / "error_analysis.md"
        
        with open(analysis_path, 'w') as f:
            f.write("# Error Analysis and Failure Modes\n\n")
            
            f.write("## Recognition Errors\n\n")
            
            # False Accept/Reject Analysis
            if 'verification_metrics' in recognition_results:
                metrics = recognition_results['verification_metrics']
                
                if 'far_at_tar' in metrics:
                    far_rates = metrics['far_at_tar']
                    f.write("### False Accept Rates at Different TAR Levels\n\n")
                    f.write("| Target TAR | FAR |\n")
                    f.write("|------------|-----|\n")
                    
                    tar_levels = [0.95, 0.99, 0.999]
                    for tar in tar_levels:
                        if tar in far_rates:
                            f.write(f"| {tar:.1%} | {far_rates[tar]:.6f} |\n")
                    
                    f.write("\n")
            
            # Attribute Prediction Errors
            f.write("## Attribute Prediction Errors\n\n")
            
            if 'problematic_attributes' in attribute_results:
                problematic = attribute_results['problematic_attributes']
                
                if problematic:
                    f.write("### Poor Performing Attributes\n\n")
                    f.write("Attributes with F1-score < 0.7:\n\n")
                    
                    for attr in problematic:
                        if 'individual_results' in attribute_results:
                            if attr in attribute_results['individual_results']:
                                metrics = attribute_results['individual_results'][attr]['metrics']
                                f1 = metrics.get('f1', 0.0)
                                precision = metrics.get('precision', 0.0)
                                recall = metrics.get('recall', 0.0)
                                
                                f.write(f"- **{attr}**: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}\n")
                            else:
                                f.write(f"- **{attr}**: Detailed metrics not available\n")
                    
                    f.write("\n")
                else:
                    f.write("No attributes with poor performance (F1 < 0.7) detected.\n\n")
            
            # Common Failure Patterns
            f.write("## Common Failure Patterns\n\n")
            f.write("Based on the evaluation results, common failure patterns may include:\n\n")
            
            failure_patterns = []
            
            # Check for demographic bias patterns
            if 'demographic_analysis' in recognition_results:
                demo_analysis = recognition_results['demographic_analysis']
                for attr, groups in demo_analysis.items():
                    aucs = [group_data.get('auc', 0) for group_data in groups.values()]
                    if max(aucs) - min(aucs) > 0.05:
                        failure_patterns.append(f"Performance varies significantly across {attr} groups")
            
            # Check for attribute correlation issues
            if 'correlation_analysis' in attribute_results:
                if 'strong_correlations' in attribute_results['correlation_analysis']:
                    strong_corrs = attribute_results['correlation_analysis']['strong_correlations']
                    if strong_corrs:
                        failure_patterns.append("Strong attribute correlations may indicate model shortcuts")
            
            if failure_patterns:
                for pattern in failure_patterns:
                    f.write(f"- {pattern}\n")
            else:
                f.write("- No obvious failure patterns detected in current evaluation\n")
            
            f.write("\n")
            
            # Recommendations for Error Reduction
            f.write("## Recommendations for Error Reduction\n\n")
            
            recommendations = [
                "Collect more diverse training data for underrepresented groups",
                "Implement data augmentation strategies for challenging conditions",
                "Consider ensemble methods for improved robustness",
                "Apply bias mitigation techniques during training",
                "Regular monitoring and evaluation on diverse test sets"
            ]
            
            for rec in recommendations:
                f.write(f"- {rec}\n")
            
            f.write("\n")
        
        return str(analysis_path)
    
    def _create_recommendations(self,
                              recognition_results: Dict[str, Any],
                              attribute_results: Dict[str, Any],
                              fairness_results: Dict[str, Any],
                              robustness_results: Dict[str, Any],
                              report_dir: Path) -> str:
        """Create recommendations for improvement"""
        
        recommendations_path = report_dir / "recommendations.md"
        
        with open(recommendations_path, 'w') as f:
            f.write("# Recommendations for System Improvement\n\n")
            
            f.write("## Priority Recommendations\n\n")
            
            priority_recs = []
            
            # High priority fairness issues
            if 'executive_summary' in fairness_results:
                bias_severity = fairness_results['executive_summary']['bias_severity']
                if bias_severity in ['High', 'Critical']:
                    priority_recs.append({
                        'priority': 'HIGH',
                        'category': 'Fairness',
                        'issue': f'{bias_severity} demographic bias detected',
                        'recommendation': 'Implement bias mitigation strategies and collect more balanced training data'
                    })
            
            # Performance issues
            if 'verification_metrics' in recognition_results:
                auc = recognition_results['verification_metrics']['auc']
                if auc < 0.9:
                    priority_recs.append({
                        'priority': 'HIGH',
                        'category': 'Performance',
                        'issue': f'Verification AUC below 0.9 ({auc:.4f})',
                        'recommendation': 'Improve model architecture, training data quality, or training procedures'
                    })
            
            # Robustness issues
            if 'critical_occlusion_threshold' in robustness_results:
                thresholds = [t for t in robustness_results['critical_occlusion_threshold'].values() if t]
                if thresholds and min(thresholds) < 0.2:
                    priority_recs.append({
                        'priority': 'MEDIUM',
                        'category': 'Robustness',
                        'issue': 'Low robustness to occlusion attacks',
                        'recommendation': 'Implement occlusion-aware training and data augmentation'
                    })
            
            # Sort by priority
            priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            priority_recs.sort(key=lambda x: priority_order[x['priority']])
            
            for i, rec in enumerate(priority_recs, 1):
                f.write(f"### {i}. {rec['category']} - {rec['priority']} Priority\n\n")
                f.write(f"**Issue:** {rec['issue']}\n\n")
                f.write(f"**Recommendation:** {rec['recommendation']}\n\n")
            
            # Specific Improvement Strategies
            f.write("## Specific Improvement Strategies\n\n")
            
            f.write("### Data Collection and Augmentation\n\n")
            f.write("- Collect more diverse training data covering edge cases\n")
            f.write("- Implement targeted data augmentation for challenging conditions\n")
            f.write("- Balance representation across demographic groups\n")
            f.write("- Include more examples of occlusion, pose variation, and illumination changes\n\n")
            
            f.write("### Model Architecture and Training\n\n")
            f.write("- Consider attention mechanisms for better feature localization\n")
            f.write("- Implement multi-task learning for improved attribute consistency\n")
            f.write("- Use adversarial training for robustness improvement\n")
            f.write("- Apply fairness-aware training techniques\n\n")
            
            f.write("### Evaluation and Monitoring\n\n")
            f.write("- Establish continuous evaluation pipeline\n")
            f.write("- Monitor performance across demographic groups\n")
            f.write("- Regular robustness testing with new attack methods\n")
            f.write("- Human evaluation of explanation quality\n\n")
            
            # Timeline and Implementation
            f.write("## Implementation Timeline\n\n")
            
            timeline_items = [
                ("Week 1-2", "Address high-priority fairness issues", "HIGH"),
                ("Week 3-4", "Implement data augmentation strategies", "MEDIUM"),
                ("Week 5-8", "Model architecture improvements", "MEDIUM"),
                ("Week 9-12", "Comprehensive testing and validation", "LOW")
            ]
            
            f.write("| Timeline | Task | Priority |\n")
            f.write("|----------|------|----------|\n")
            
            for timeline, task, priority in timeline_items:
                f.write(f"| {timeline} | {task} | {priority} |\n")
            
            f.write("\n")
        
        return str(recommendations_path)
    
    def _create_technical_appendix(self,
                                 model_info: Dict[str, Any],
                                 dataset_info: Dict[str, Any],
                                 report_dir: Path) -> str:
        """Create technical appendix with model and dataset details"""
        
        appendix_path = report_dir / "technical_appendix.md"
        
        with open(appendix_path, 'w') as f:
            f.write("# Technical Appendix\n\n")
            
            f.write("## Model Information\n\n")
            
            if model_info:
                f.write("| Property | Value |\n")
                f.write("|----------|-------|\n")
                
                for key, value in model_info.items():
                    f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
            else:
                f.write("Model information not provided.\n")
            
            f.write("\n## Dataset Information\n\n")
            
            if dataset_info:
                f.write("| Property | Value |\n")
                f.write("|----------|-------|\n")
                
                for key, value in dataset_info.items():
                    f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
            else:
                f.write("Dataset information not provided.\n")
            
            f.write("\n## Evaluation Methodology\n\n")
            
            f.write("### Recognition Evaluation\n\n")
            f.write("- **Verification**: ROC curve analysis, AUC, EER, TAR@FAR\n")
            f.write("- **Identification**: Rank-N accuracy with bootstrap confidence intervals\n")
            f.write("- **Demographics**: Performance disaggregation across protected attributes\n\n")
            
            f.write("### Attribute Evaluation\n\n")
            f.write("- **Binary attributes**: Precision, recall, F1-score, accuracy\n")
            f.write("- **Categorical attributes**: Multi-class classification metrics\n")
            f.write("- **Correlation analysis**: Pearson correlation between attributes\n\n")
            
            f.write("### Fairness Evaluation\n\n")
            f.write("- **Group gaps**: TPR/FPR differences between demographic groups\n")
            f.write("- **Statistical testing**: Permutation tests for significance\n")
            f.write("- **Intersectional analysis**: Performance across multiple attributes\n\n")
            
            f.write("### Robustness Evaluation\n\n")
            f.write("- **Occlusion attacks**: Random, targeted, and semantic occlusions\n")
            f.write("- **Pose variations**: Yaw, pitch, roll transformations\n")
            f.write("- **Illumination changes**: Brightness and contrast variations\n")
            f.write("- **Domain shift**: Cross-dataset evaluation\n\n")
            
            f.write("### Explainability Evaluation\n\n")
            f.write("- **Sanity checks**: Model dependence testing\n")
            f.write("- **Fidelity tests**: Deletion/insertion curves\n")
            f.write("- **Localization**: Face region focus analysis\n")
            f.write("- **Human evaluation**: Structured assessment framework\n\n")
        
        return str(appendix_path)
    
    def _create_master_report(self,
                            report_metadata: Dict[str, Any],
                            recognition_results: Dict[str, Any],
                            attribute_results: Dict[str, Any],
                            explanation_results: Dict[str, Any],
                            fairness_results: Dict[str, Any],
                            robustness_results: Dict[str, Any],
                            model_info: Dict[str, Any],
                            dataset_info: Dict[str, Any],
                            report_dir: Path) -> str:
        """Create master HTML report combining all sections"""
        
        html_path = report_dir / "master_report.html"
        
        # Read all markdown files and convert to HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.report_name} - Comprehensive Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
                h3 {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .warning {{ background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; }}
                .success {{ background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; }}
                .error {{ background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{self.report_name} - Comprehensive Evaluation Report</h1>
            <p><strong>Generated:</strong> {self.timestamp}</p>
            
            <div class="summary">
                <h2>Quick Summary</h2>
                <p>This report provides a comprehensive evaluation of the face recognition system across
                multiple dimensions including performance, fairness, robustness, and explainability.</p>
            </div>
        """
        
        # Add navigation
        html_content += """
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#recognition-analysis">Recognition Performance</a></li>
                <li><a href="#attribute-analysis">Attribute Analysis</a></li>
                <li><a href="#explanation-analysis">Explainability Evaluation</a></li>
                <li><a href="#fairness-analysis">Fairness Assessment</a></li>
                <li><a href="#robustness-analysis">Robustness Evaluation</a></li>
                <li><a href="#error-analysis">Error Analysis</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
                <li><a href="#technical-appendix">Technical Appendix</a></li>
            </ul>
        """
        
        # Include content from each section (simplified)
        sections = [
            ("executive-summary", "Executive Summary"),
            ("recognition-analysis", "Recognition Performance Analysis"),
            ("attribute-analysis", "Attribute Analysis"),
            ("explanation-analysis", "Explainability Evaluation"),
            ("fairness-analysis", "Fairness Assessment"),
            ("robustness-analysis", "Robustness Evaluation"),
            ("error-analysis", "Error Analysis"),
            ("recommendations", "Recommendations"),
            ("technical-appendix", "Technical Appendix")
        ]
        
        for section_id, section_title in sections:
            html_content += f"""
                <div id="{section_id}">
                    <h2>{section_title}</h2>
                    <p>Detailed {section_title.lower()} content would be embedded here in a full implementation.</p>
                    <p><em>See individual files in the report directory for complete analysis.</em></p>
                </div>
            """
        
        html_content += """
            <footer>
                <hr>
                <p><em>Report generated by Face Recognition Evaluation System</em></p>
            </footer>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)