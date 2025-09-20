"""
Evaluation Metrics Package
=========================

Comprehensive evaluation framework for face recognition systems including:
- Recognition metrics (verification, identification)
- Attribute accuracy metrics  
- Explanation fidelity metrics
- Fairness and bias evaluation
- Robustness testing
"""

from .recognition_metrics import RecognitionEvaluator
from .attribute_metrics import AttributeEvaluator
from .explanation_metrics import ExplanationEvaluator
from .fairness_metrics import FairnessEvaluator
from .robustness_metrics import RobustnessEvaluator
from .report_generator import EvaluationReportGenerator

__all__ = [
    'RecognitionEvaluator',
    'AttributeEvaluator', 
    'ExplanationEvaluator',
    'FairnessEvaluator',
    'RobustnessEvaluator',
    'EvaluationReportGenerator'
]