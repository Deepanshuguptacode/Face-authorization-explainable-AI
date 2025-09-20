"""
Test Explainability Module
==========================

Test script to validate all explainability methods and generate example outputs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from src.models.face_recognition_model import create_model, MODEL_CONFIGS
from src.explainability import (
    ExplainabilityPipeline, 
    GradCAMExplainer,
    AttributeExplainer,
    PrototypeExplainer,
    TextualExplainer
)


def create_test_images(batch_size=2):
    """Create test images for explanation"""
    # Create synthetic face-like images
    images = []
    for i in range(batch_size):
        # Create a simple face-like pattern
        img = np.zeros((224, 224, 3), dtype=np.float32)
        
        # Add some face-like features
        # Face oval
        cv2.ellipse(img, (112, 112), (80, 100), 0, 0, 360, (0.8, 0.7, 0.6), -1)
        
        # Eyes
        cv2.circle(img, (90, 90), 10, (0.2, 0.2, 0.2), -1)  # Left eye
        cv2.circle(img, (134, 90), 10, (0.2, 0.2, 0.2), -1)  # Right eye
        
        # Nose
        cv2.line(img, (112, 100), (112, 130), (0.6, 0.5, 0.4), 2)
        
        # Mouth
        cv2.ellipse(img, (112, 150), (20, 10), 0, 0, 180, (0.8, 0.4, 0.4), 2)
        
        # Add some variation for different images
        if i % 2 == 1:
            # Add glasses
            cv2.rectangle(img, (70, 80), (100, 100), (0.1, 0.1, 0.1), 2)
            cv2.rectangle(img, (124, 80), (154, 100), (0.1, 0.1, 0.1), 2)
            cv2.line(img, (100, 90), (124, 90), (0.1, 0.1, 0.1), 2)
        
        # Convert to CHW format for PyTorch
        img = np.transpose(img, (2, 0, 1))
        images.append(img)
    
    return torch.tensor(np.array(images), dtype=torch.float32)


def test_individual_explainers():
    """Test individual explainer components"""
    print("="*60)
    print("TESTING INDIVIDUAL EXPLAINER COMPONENTS")
    print("="*60)
    
    # Create test model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_config = MODEL_CONFIGS['baseline']
    model = create_model(model_config)
    model.to(device)
    model.eval()
    
    # Create test images
    test_images = create_test_images(4).to(device)
    print(f"Created test images: {test_images.shape}")
    
    # Test Grad-CAM Explainer
    print("\n1. Testing Grad-CAM Explainer...")
    try:
        gradcam_explainer = GradCAMExplainer(model, device=device)
        gradcam_result = gradcam_explainer.explain(
            test_images[0:1], 
            target_type='identity'
        )
        print(f"✓ Grad-CAM explanation generated")
        print(f"  - Grad-CAM shape: {gradcam_result['grad_cam'].shape}")
        print(f"  - Target score: {gradcam_result['target_score']:.4f}")
    except Exception as e:
        print(f"✗ Grad-CAM failed: {e}")
    
    # Test Attribute Explainer
    print("\n2. Testing Attribute Explainer...")
    try:
        attr_explainer = AttributeExplainer(model, device=device)
        attr_result = attr_explainer.explain(test_images[0:1])
        print(f"✓ Attribute explanation generated")
        print(f"  - Attributes analyzed: {len(attr_result['attribute_predictions'])}")
        print(f"  - Top positive: {len(attr_result['top_positive_attributes'])}")
        print(f"  - Top negative: {len(attr_result['top_negative_attributes'])}")
    except Exception as e:
        print(f"✗ Attribute explanation failed: {e}")
    
    # Test verification with attributes
    print("\n3. Testing Attribute Verification...")
    try:
        verification_result = attr_explainer.explain_verification(
            test_images[0:1], test_images[1:2], similarity_threshold=0.5
        )
        print(f"✓ Attribute verification explanation generated")
        print(f"  - Similarity: {verification_result['similarity']:.4f}")
        print(f"  - Is match: {verification_result['is_match']}")
        print(f"  - Agreement rate: {verification_result['agreement_analysis']['agreement_rate']:.2%}")
    except Exception as e:
        print(f"✗ Attribute verification failed: {e}")
    
    # Test Prototype Explainer (basic functionality)
    print("\n4. Testing Prototype Explainer...")
    try:
        proto_explainer = PrototypeExplainer(model, device=device)
        
        # Build a small prototype database
        identities = torch.randint(0, 10, (4,))
        database_info = proto_explainer.build_prototype_database(
            test_images, identities, n_prototypes_per_identity=2
        )
        print(f"✓ Prototype database built")
        print(f"  - Total prototypes: {database_info['total_prototypes']}")
        print(f"  - Unique identities: {database_info['unique_identities']}")
        
        # Test finding nearest prototypes
        with torch.no_grad():
            query_embedding = model.get_embeddings(test_images[0:1]).cpu().numpy()[0]
        
        nearest = proto_explainer.find_nearest_prototypes(query_embedding, k=3)
        print(f"✓ Found {len(nearest)} nearest prototypes")
        
    except Exception as e:
        print(f"✗ Prototype explanation failed: {e}")
    
    # Test Textual Explainer
    print("\n5. Testing Textual Explainer...")
    try:
        textual_explainer = TextualExplainer(model, device=device, explanation_style='comprehensive')
        
        # Test identity explanation
        identity_explanation = textual_explainer.explain_identity(
            test_images[0:1],
            predicted_identity=5,
            attribute_explanation=attr_result if 'attr_result' in locals() else None
        )
        print(f"✓ Identity textual explanation generated")
        print(f"  - Explanation: {identity_explanation['explanation'][:100]}...")
        print(f"  - Confidence: {identity_explanation['confidence']:.4f}")
        
        # Test verification explanation
        verification_text = textual_explainer.explain_verification(
            test_images[0:1], test_images[1:2], threshold=0.5,
            attribute_explanation=verification_result if 'verification_result' in locals() else None
        )
        print(f"✓ Verification textual explanation generated")
        print(f"  - Explanation: {verification_text['explanation'][:100]}...")
        
    except Exception as e:
        print(f"✗ Textual explanation failed: {e}")


def test_explainability_pipeline():
    """Test the complete explainability pipeline"""
    print("\n" + "="*60)
    print("TESTING COMPLETE EXPLAINABILITY PIPELINE")
    print("="*60)
    
    # Create test model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = MODEL_CONFIGS['baseline']
    model = create_model(model_config)
    model.to(device)
    model.eval()
    
    # Create test images
    test_images = create_test_images(4)
    
    # Initialize explainability pipeline
    print("\n1. Initializing Explainability Pipeline...")
    try:
        explanation_methods = ['gradcam', 'attributes', 'textual']  # Start with basic methods
        pipeline = ExplainabilityPipeline(
            model, 
            device=device,
            explanation_methods=explanation_methods,
            explanation_style='comprehensive'
        )
        print(f"✓ Pipeline initialized with methods: {explanation_methods}")
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return
    
    # Test identity explanation
    print("\n2. Testing Complete Identity Explanation...")
    try:
        identity_result = pipeline.explain_identity(
            test_images[0:1],
            include_visualizations=True
        )
        print(f"✓ Complete identity explanation generated")
        print(f"  - Embedding shape: {identity_result.embedding.shape}")
        print(f"  - Has Grad-CAM: {identity_result.grad_cam is not None}")
        print(f"  - Has attributes: {identity_result.predicted_attributes is not None}")
        print(f"  - Textual explanation: {identity_result.textual_explanation[:150] if identity_result.textual_explanation else 'None'}...")
        
        # Check visualizations
        if hasattr(identity_result, 'detailed_results') and 'visualizations' in identity_result.detailed_results:
            viz = identity_result.detailed_results['visualizations']
            print(f"  - Visualizations created: {list(viz.keys())}")
        
    except Exception as e:
        print(f"✗ Identity explanation failed: {e}")
    
    # Test verification explanation
    print("\n3. Testing Complete Verification Explanation...")
    try:
        verification_result = pipeline.explain_verification(
            test_images[0:1],
            test_images[1:2],
            threshold=0.5,
            include_visualizations=True
        )
        print(f"✓ Complete verification explanation generated")
        print(f"  - Match score: {verification_result.match_score:.4f}")
        print(f"  - Has Grad-CAM: {verification_result.grad_cam is not None}")
        print(f"  - Has attributes: {verification_result.predicted_attributes is not None}")
        print(f"  - Textual explanation: {verification_result.textual_explanation[:150] if verification_result.textual_explanation else 'None'}...")
        
        # Check detailed results
        if hasattr(verification_result, 'detailed_results'):
            methods_used = list(verification_result.detailed_results.keys())
            print(f"  - Explanation methods used: {methods_used}")
        
    except Exception as e:
        print(f"✗ Verification explanation failed: {e}")
    
    # Test with prototypes (if we have them)
    print("\n4. Testing with Prototype Database...")
    try:
        # Build prototype database
        identities = torch.randint(0, 5, (4,))
        database_info = pipeline.build_prototype_database(test_images, identities)
        print(f"✓ Prototype database built: {database_info}")
        
        # Enable prototypes and test again
        pipeline.enabled_methods.append('prototypes')
        pipeline._initialize_explainers()
        
        proto_identity_result = pipeline.explain_identity(test_images[0:1])
        print(f"✓ Identity explanation with prototypes generated")
        print(f"  - Has similar prototypes: {proto_identity_result.similar_prototypes is not None}")
        
    except Exception as e:
        print(f"✗ Prototype testing failed: {e}")


def test_explanation_styles():
    """Test different explanation styles"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT EXPLANATION STYLES")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = MODEL_CONFIGS['baseline']
    model = create_model(model_config)
    model.to(device)
    model.eval()
    
    test_images = create_test_images(2).to(device)
    
    for style in ['brief', 'comprehensive', 'technical']:
        print(f"\n{style.upper()} Style:")
        try:
            pipeline = ExplainabilityPipeline(
                model, 
                device=device,
                explanation_methods=['gradcam', 'attributes', 'textual'],
                explanation_style=style
            )
            
            result = pipeline.explain_verification(
                test_images[0:1], test_images[1:2], threshold=0.5
            )
            
            print(f"✓ {style} explanation:")
            print(f"  {result.textual_explanation}")
            
        except Exception as e:
            print(f"✗ {style} style failed: {e}")


def save_example_outputs():
    """Save example explanation outputs"""
    print("\n" + "="*60)
    print("SAVING EXAMPLE OUTPUTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = MODEL_CONFIGS['baseline']
    model = create_model(model_config)
    model.to(device)
    model.eval()
    
    test_images = create_test_images(2).to(device)
    
    # Create output directory
    output_dir = "explainability_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pipeline = ExplainabilityPipeline(
            model, 
            device=device,
            explanation_methods=['gradcam', 'attributes', 'textual'],
            explanation_style='comprehensive'
        )
        
        # Generate and save identity explanation
        identity_result = pipeline.explain_identity(test_images[0:1], include_visualizations=True)
        
        # Save original image
        original_img = test_images[0].permute(1, 2, 0).numpy()
        original_img = (original_img * 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/original_image.jpg", cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        
        # Save Grad-CAM if available
        if identity_result.grad_cam is not None:
            gradcam_img = (identity_result.grad_cam * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/gradcam_heatmap.jpg", gradcam_img)
        
        # Save textual explanation
        with open(f"{output_dir}/explanation.txt", "w") as f:
            f.write("IDENTITY EXPLANATION\n")
            f.write("="*50 + "\n\n")
            f.write(identity_result.textual_explanation or "No textual explanation generated")
            f.write("\n\n")
            
            if identity_result.predicted_attributes:
                f.write("PREDICTED ATTRIBUTES\n")
                f.write("-"*30 + "\n")
                for attr, pred in list(identity_result.predicted_attributes.items())[:10]:
                    conf = identity_result.attribute_confidences.get(attr, 0.0) if identity_result.attribute_confidences else 0.0
                    status = "Present" if pred == 1 else "Absent"
                    f.write(f"{attr.replace('_', ' ')}: {status} (confidence: {conf:.3f})\n")
        
        print(f"✓ Example outputs saved to {output_dir}/")
        
    except Exception as e:
        print(f"✗ Saving examples failed: {e}")


def main():
    """Run all explainability tests"""
    print("EXPLAINABILITY MODULE TEST SUITE")
    print("=" * 80)
    
    try:
        # Test individual components
        test_individual_explainers()
        
        # Test complete pipeline
        test_explainability_pipeline()
        
        # Test different styles
        test_explanation_styles()
        
        # Save examples
        save_example_outputs()
        
        print("\n" + "="*80)
        print("ALL EXPLAINABILITY TESTS COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()