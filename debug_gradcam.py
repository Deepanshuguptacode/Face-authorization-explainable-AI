"""
Debug GradCAM issue
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.models.face_recognition_model import create_model, MODEL_CONFIGS
from src.explainability.visual_explanations import GradCAMExplainer

# Create model
model = create_model(MODEL_CONFIGS['baseline'])
print("Model created")

# Create explainer
explainer = GradCAMExplainer(model, device='cpu')
print("Explainer created")

# Check if attribute exists
print(f"Has target_layer: {hasattr(explainer, 'target_layer')}")
print(f"target_layer value: {getattr(explainer, 'target_layer', 'NOT FOUND')}")

# Create test image
test_image = torch.randn(1, 3, 224, 224)
print("Test image created")

# Try to explain
try:
    result = explainer.explain(test_image, target_type='identity')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()