"""
Test Script for Face Recognition Pipeline
=========================================

This script tests all components of the face recognition pipeline:
1. Model architecture
2. Loss functions  
3. Data loading
4. Training framework
5. Evaluation

Run this to verify everything is working before starting full experiments.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_architecture():
    """Test model creation and forward pass"""
    print("Testing model architecture...")
    
    try:
        from models.face_recognition_model import MODEL_CONFIGS, create_model
        
        # Create model
        config = MODEL_CONFIGS['baseline'].copy()
        config['num_classes'] = 100  # Small number for testing
        model = create_model(config)
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, config['num_classes'], (batch_size,))
        
        model.train()
        output = model(x, labels, return_embeddings=True)
        
        print(f"✓ Model forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Identity logits: {output['identity_logits'].shape}")
        print(f"  - Attribute logits: {output['attribute_logits'].shape}")
        print(f"  - Embeddings: {output['embeddings'].shape}")
        
        # Test inference
        model.eval()
        embeddings = model.extract_embeddings(x)
        print(f"  - Extracted embeddings: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss function computation"""
    print("\\nTesting loss functions...")
    
    try:
        from models.losses import LOSS_CONFIGS, create_loss_function
        
        # Create dummy data
        batch_size = 8
        num_classes = 100
        num_attributes = 40
        
        identity_logits = torch.randn(batch_size, num_classes)
        attribute_logits = torch.randn(batch_size, num_attributes)
        identity_labels = torch.randint(0, num_classes, (batch_size,))
        attribute_labels = torch.randint(0, 2, (batch_size, num_attributes)) * 2 - 1  # {-1, 1}
        
        # Create loss function
        config = LOSS_CONFIGS['baseline']
        loss_fn = create_loss_function(config, attribute_labels)
        
        # Test forward pass
        losses = loss_fn(
            identity_logits=identity_logits,
            attribute_logits=attribute_logits,
            identity_labels=identity_labels,
            attribute_labels=attribute_labels
        )
        
        print(f"✓ Loss computation successful")
        for key, value in losses.items():
            if torch.is_tensor(value):
                print(f"  - {key}: {value.item():.4f}")
            else:
                print(f"  - {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading pipeline"""
    print("\\nTesting data loading...")
    
    try:
        from data.dataset import DATA_CONFIGS, create_dataloaders
        
        # Create test config with smaller batch size
        config = DATA_CONFIGS['baseline'].copy()
        config['batch_size'] = 4
        config['num_workers'] = 0  # Avoid multiprocessing issues in test
        
        # Create data loaders (this will test if the processed data exists)
        train_loader, val_loader, test_loader, identity_encoding = create_dataloaders(config)
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Train: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        print(f"  - Val: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
        print(f"  - Test: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        print(f"  - Unique identities: {len(identity_encoding)}")
        
        # Test loading a batch
        try:
            batch = next(iter(train_loader))
            print(f"✓ Sample batch loaded successfully")
            print(f"  - Image shape: {batch['image'].shape}")
            
            if 'identity' in batch:
                print(f"  - Identity shape: {batch['identity'].shape}")
                print(f"  - Identity range: [{batch['identity'].min()}, {batch['identity'].max()}]")
            
            if 'attributes' in batch:
                print(f"  - Attributes shape: {batch['attributes'].shape}")
                print(f"  - Attributes range: [{batch['attributes'].min():.1f}, {batch['attributes'].max():.1f}]")
            
            return True
            
        except Exception as e:
            print(f"✗ Batch loading failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_framework():
    """Test training framework setup"""
    print("\\nTesting training framework...")
    
    try:
        from models.face_recognition_model import MODEL_CONFIGS
        from models.losses import LOSS_CONFIGS
        from data.dataset import DATA_CONFIGS
        from training.trainer import TRAIN_CONFIGS, create_trainer
        
        # Create test configs
        model_config = MODEL_CONFIGS['baseline'].copy()
        loss_config = LOSS_CONFIGS['baseline'].copy()
        data_config = DATA_CONFIGS['baseline'].copy()
        train_config = TRAIN_CONFIGS['fast_test'].copy()
        
        # Reduce batch size and workers for testing
        data_config['batch_size'] = 4
        data_config['num_workers'] = 0
        train_config['num_epochs'] = 1
        train_config['experiment_name'] = 'test_run'
        
        # Create trainer
        trainer, train_loader, val_loader, test_loader = create_trainer(
            model_config=model_config,
            loss_config=loss_config,
            data_config=data_config,
            train_config=train_config
        )
        
        print(f"✓ Trainer created successfully")
        print(f"  - Model: {trainer.model.__class__.__name__}")
        print(f"  - Device: {trainer.device}")
        print(f"  - Data loaders: {len(train_loader)}/{len(val_loader)}/{len(test_loader)} batches")
        
        # Test a single forward pass
        batch = next(iter(train_loader))
        trainer.model.train()
        
        images = batch['image'].to(trainer.device)
        outputs = trainer.model(images)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {images.shape}")
        print(f"  - Identity logits: {outputs['identity_logits'].shape}")
        print(f"  - Attribute logits: {outputs['attribute_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functionality"""
    print("\\nTesting evaluation...")
    
    try:
        from evaluation.evaluator import FaceRecognitionEvaluator
        from models.face_recognition_model import create_model, MODEL_CONFIGS
        
        # Create a simple model for testing
        config = MODEL_CONFIGS['baseline'].copy()
        config['num_classes'] = 100
        model = create_model(config)
        
        # Create evaluator
        evaluator = FaceRecognitionEvaluator(model)
        
        print(f"✓ Evaluator created successfully")
        
        # Test explanation generation
        test_image = torch.randn(1, 3, 224, 224)
        explanation = evaluator.generate_explanation(test_image)
        
        print(f"✓ Explanation generated successfully")
        print(f"  - Embeddings shape: {explanation['embeddings'].shape}")
        print(f"  - Top positive attributes: {len(explanation['top_positive_attributes'])}")
        print(f"  - Top negative attributes: {len(explanation['top_negative_attributes'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("FACE RECOGNITION PIPELINE TEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("="*60)
    
    # Run tests
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Loss Functions", test_loss_functions),
        ("Data Loading", test_data_loading),
        ("Training Framework", test_training_framework),
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<20}: {status}")
        if success:
            passed += 1
    
    print(f"\\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\\n🎉 All tests passed! The pipeline is ready for training.")
    else:
        print("\\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    main()