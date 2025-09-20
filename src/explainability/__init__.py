"""
Explainability Module for Face Recognition
==========================================

This module provides comprehensive explanation capabilities for face recognition
and verification decisions, including:

1. Visual explanations (Grad-CAM, Integrated Gradients)
2. Attribute-based explanations 
3. Concept activation analysis (TCAV)
4. Prototype-based explanations
5. Counterfactual analysis
6. Textual explanation generation

The module is designed to provide interpretable insights into model decisions
for both identity verification and attribute prediction tasks.
"""

from .visual_explanations import GradCAMExplainer, IntegratedGradientsExplainer
from .attribute_explanations import AttributeExplainer
from .concept_analysis import TCAVAnalyzer
from .prototype_explanations import PrototypeExplainer
from .counterfactual_explanations import CounterfactualExplainer
from .textual_explanations import TextualExplainer
from .explainability_pipeline import ExplainabilityPipeline

__all__ = [
    'GradCAMExplainer',
    'IntegratedGradientsExplainer', 
    'AttributeExplainer',
    'TCAVAnalyzer',
    'PrototypeExplainer',
    'CounterfactualExplainer',
    'TextualExplainer',
    'ExplainabilityPipeline'
]