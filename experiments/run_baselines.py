"""
Baseline Experiment Runner
==========================

This script runs the three baseline experiments:
1. Identity-only (ArcFace) - measure verification performance
2. Multi-task (identity + attributes) - measure if attribute supervision helps embeddings
3. Attribute-only heads - for attribute accuracy

Usage:
    python run_baselines.py --experiment identity_only --epochs 50
    python run_baselines.py --experiment multi_task --epochs 50
    python run_baselines.py --experiment attributes_only --epochs 50
    python run_baselines.py --experiment all --epochs 20  # Run all experiments
"""

import argparse
import os
import sys
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.face_recognition_model import MODEL_CONFIGS, create_model
from models.losses import LOSS_CONFIGS, create_loss_function
from data.dataset import DATA_CONFIGS, create_dataloaders
from training.trainer import TRAIN_CONFIGS, create_trainer
from evaluation.evaluator import evaluate_model


def run_identity_only_experiment(epochs: int = 50, use_wandb: bool = False) -> Dict:
    """
    Run identity-only baseline experiment
    
    Args:
        epochs: Number of training epochs
        use_wandb: Use Weights & Biases logging
    
    Returns:
        Experiment results
    """
    print("="*60)
    print("BASELINE EXPERIMENT 1: IDENTITY-ONLY (ARCFACE)")
    print("="*60)
    
    # Configuration
    model_config = MODEL_CONFIGS['baseline'].copy()
    loss_config = LOSS_CONFIGS['baseline'].copy()
    data_config = DATA_CONFIGS['identity_only'].copy()  # No attributes
    train_config = TRAIN_CONFIGS['baseline'].copy()
    
    # Update configs
    train_config.update({
        'num_epochs': epochs,
        'experiment_name': 'identity_only',
        'use_wandb': use_wandb,
        'checkpoint_dir': 'experiments/checkpoints',
        'log_dir': 'experiments/logs'
    })
    
    # Create trainer
    trainer, train_loader, val_loader, test_loader = create_trainer(
        model_config=model_config,
        loss_config=loss_config,
        data_config=data_config,
        train_config=train_config
    )
    
    print(f"Model: {trainer.model.__class__.__name__}")
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")
    
    # Training
    print("\\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        train_identity=True,
        train_attributes=False,  # Identity only
        validate_every=5,
        save_every=10
    )
    
    # Evaluation
    print("\\nEvaluating model...")
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(train_config['checkpoint_dir'], 'identity_only_best.pth')
    if os.path.exists(best_checkpoint_path):
        trainer.load_checkpoint(best_checkpoint_path, load_optimizer=False)
    
    # Evaluate on test set
    evaluation_results = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        save_results=True,
        results_dir='experiments/results/identity_only'
    )
    
    results = {
        'experiment': 'identity_only',
        'training_history': history,
        'evaluation': evaluation_results,
        'config': {
            'model': model_config,
            'loss': loss_config,
            'data': data_config,
            'training': train_config
        }
    }
    
    print("✓ Identity-only experiment completed!")
    return results


def run_multi_task_experiment(epochs: int = 50, use_wandb: bool = False) -> Dict:
    """
    Run multi-task baseline experiment
    
    Args:
        epochs: Number of training epochs
        use_wandb: Use Weights & Biases logging
    
    Returns:
        Experiment results
    """
    print("="*60)
    print("BASELINE EXPERIMENT 2: MULTI-TASK (IDENTITY + ATTRIBUTES)")
    print("="*60)
    
    # Configuration
    model_config = MODEL_CONFIGS['baseline'].copy()
    loss_config = LOSS_CONFIGS['baseline'].copy()
    data_config = DATA_CONFIGS['baseline'].copy()  # Both identity and attributes
    train_config = TRAIN_CONFIGS['baseline'].copy()
    
    # Update configs
    train_config.update({
        'num_epochs': epochs,
        'experiment_name': 'multi_task',
        'use_wandb': use_wandb,
        'checkpoint_dir': 'experiments/checkpoints',
        'log_dir': 'experiments/logs'
    })
    
    # Create trainer
    trainer, train_loader, val_loader, test_loader = create_trainer(
        model_config=model_config,
        loss_config=loss_config,
        data_config=data_config,
        train_config=train_config
    )
    
    print(f"Model: {trainer.model.__class__.__name__}")
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")
    
    # Training
    print("\\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        train_identity=True,
        train_attributes=True,  # Multi-task
        validate_every=5,
        save_every=10
    )
    
    # Evaluation
    print("\\nEvaluating model...")
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(train_config['checkpoint_dir'], 'multi_task_best.pth')
    if os.path.exists(best_checkpoint_path):
        trainer.load_checkpoint(best_checkpoint_path, load_optimizer=False)
    
    # Evaluate on test set
    evaluation_results = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        save_results=True,
        results_dir='experiments/results/multi_task'
    )
    
    results = {
        'experiment': 'multi_task',
        'training_history': history,
        'evaluation': evaluation_results,
        'config': {
            'model': model_config,
            'loss': loss_config,
            'data': data_config,
            'training': train_config
        }
    }
    
    print("✓ Multi-task experiment completed!")
    return results


def run_attributes_only_experiment(epochs: int = 50, use_wandb: bool = False) -> Dict:
    """
    Run attributes-only baseline experiment
    
    Args:
        epochs: Number of training epochs
        use_wandb: Use Weights & Biases logging
    
    Returns:
        Experiment results
    """
    print("="*60)
    print("BASELINE EXPERIMENT 3: ATTRIBUTES-ONLY")
    print("="*60)
    
    # Configuration
    model_config = MODEL_CONFIGS['attributes_only'].copy()
    loss_config = LOSS_CONFIGS['baseline'].copy()
    data_config = DATA_CONFIGS['attributes_only'].copy()  # No identity
    train_config = TRAIN_CONFIGS['baseline'].copy()
    
    # Update configs
    train_config.update({
        'num_epochs': epochs,
        'experiment_name': 'attributes_only',
        'use_wandb': use_wandb,
        'checkpoint_dir': 'experiments/checkpoints',
        'log_dir': 'experiments/logs'
    })
    
    # Create trainer
    trainer, train_loader, val_loader, test_loader = create_trainer(
        model_config=model_config,
        loss_config=loss_config,
        data_config=data_config,
        train_config=train_config
    )
    
    print(f"Model: {trainer.model.__class__.__name__}")
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")
    
    # Training
    print("\\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        train_identity=False,  # Attributes only
        train_attributes=True,
        validate_every=5,
        save_every=10
    )
    
    # Evaluation
    print("\\nEvaluating model...")
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(train_config['checkpoint_dir'], 'attributes_only_best.pth')
    if os.path.exists(best_checkpoint_path):
        trainer.load_checkpoint(best_checkpoint_path, load_optimizer=False)
    
    # Evaluate on test set
    evaluation_results = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        save_results=True,
        results_dir='experiments/results/attributes_only'
    )
    
    results = {
        'experiment': 'attributes_only',
        'training_history': history,
        'evaluation': evaluation_results,
        'config': {
            'model': model_config,
            'loss': loss_config,
            'data': data_config,
            'training': train_config
        }
    }
    
    print("✓ Attributes-only experiment completed!")
    return results


def compare_experiments(results_list: List[Dict]) -> pd.DataFrame:
    """
    Compare results from multiple experiments
    
    Args:
        results_list: List of experiment results
    
    Returns:
        Comparison DataFrame
    """
    import pandas as pd
    
    comparison_data = []
    
    for results in results_list:
        experiment_name = results['experiment']
        history = results['training_history']
        evaluation = results['evaluation']
        
        # Extract key metrics
        row = {
            'experiment': experiment_name,
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
        }
        
        # Identity metrics
        if 'train_identity_acc' in history:
            row['final_train_identity_acc'] = history['train_identity_acc'][-1] if history['train_identity_acc'] else None
        if 'val_identity_acc' in history:
            row['final_val_identity_acc'] = history['val_identity_acc'][-1] if history['val_identity_acc'] else None
        
        # Attribute metrics
        if 'train_attribute_acc' in history:
            row['final_train_attribute_acc'] = history['train_attribute_acc'][-1] if history['train_attribute_acc'] else None
        if 'val_attribute_acc' in history:
            row['final_val_attribute_acc'] = history['val_attribute_acc'][-1] if history['val_attribute_acc'] else None
        
        # Test evaluation metrics
        if 'attributes' in evaluation and 'overall_accuracy' in evaluation['attributes']:
            row['test_attribute_accuracy'] = evaluation['attributes']['overall_accuracy']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def save_experiment_summary(results_list: List[Dict], output_dir: str = 'experiments/results'):
    """
    Save comprehensive experiment summary
    
    Args:
        results_list: List of experiment results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    for results in results_list:
        experiment_name = results['experiment']
        
        # Save full results
        results_file = os.path.join(output_dir, f'{experiment_name}_full_results.json')
        with open(results_file, 'w') as f:
            # Convert to JSON-serializable format
            serializable_results = {}
            for key, value in results.items():
                if key == 'training_history':
                    serializable_results[key] = {
                        k: v for k, v in value.items() 
                        if isinstance(v, (list, float, int, str))
                    }
                elif key == 'evaluation':
                    # Skip complex evaluation results for now
                    serializable_results[key] = 'saved_separately'
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
    
    # Create comparison
    comparison_df = compare_experiments(results_list)
    comparison_file = os.path.join(output_dir, 'experiments_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("FACE RECOGNITION BASELINE EXPERIMENTS SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        for results in results_list:
            experiment_name = results['experiment']
            f.write(f"Experiment: {experiment_name.upper()}\\n")
            f.write("-" * 30 + "\\n")
            
            history = results['training_history']
            if history:
                f.write(f"Training epochs: {len(history.get('train_loss', []))}\\n")
                if 'train_loss' in history and history['train_loss']:
                    f.write(f"Final train loss: {history['train_loss'][-1]:.4f}\\n")
                if 'val_loss' in history and history['val_loss']:
                    f.write(f"Final val loss: {history['val_loss'][-1]:.4f}\\n")
                
                if 'train_identity_acc' in history and history['train_identity_acc']:
                    f.write(f"Final train identity acc: {history['train_identity_acc'][-1]:.4f}\\n")
                if 'val_identity_acc' in history and history['val_identity_acc']:
                    f.write(f"Final val identity acc: {history['val_identity_acc'][-1]:.4f}\\n")
                    
                if 'train_attribute_acc' in history and history['train_attribute_acc']:
                    f.write(f"Final train attribute acc: {history['train_attribute_acc'][-1]:.4f}\\n")
                if 'val_attribute_acc' in history and history['val_attribute_acc']:
                    f.write(f"Final val attribute acc: {history['val_attribute_acc'][-1]:.4f}\\n")
            
            f.write("\\n")
    
    print(f"\\nExperiment summary saved to: {output_dir}")
    print(f"- Individual results: *_full_results.json")
    print(f"- Comparison table: experiments_comparison.csv")
    print(f"- Summary report: experiment_summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Run baseline face recognition experiments')
    parser.add_argument('--experiment', type=str, 
                       choices=['identity_only', 'multi_task', 'attributes_only', 'all'],
                       default='all',
                       help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Create output directories
    os.makedirs('experiments/checkpoints', exist_ok=True)
    os.makedirs('experiments/logs', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)
    
    # Run experiments
    results_list = []
    
    if args.experiment == 'all':
        experiments = ['identity_only', 'multi_task', 'attributes_only']
    else:
        experiments = [args.experiment]
    
    for exp_name in experiments:
        print(f"\\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
        print(f"{'='*80}")
        
        try:
            if exp_name == 'identity_only':
                results = run_identity_only_experiment(args.epochs, args.use_wandb)
            elif exp_name == 'multi_task':
                results = run_multi_task_experiment(args.epochs, args.use_wandb)
            elif exp_name == 'attributes_only':
                results = run_attributes_only_experiment(args.epochs, args.use_wandb)
            
            results_list.append(results)
            print(f"✓ {exp_name} experiment completed successfully!")
            
        except Exception as e:
            print(f"✗ {exp_name} experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive summary
    if results_list:
        save_experiment_summary(results_list)
        
        # Print final comparison
        print(f"\\n{'='*80}")
        print("EXPERIMENT COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = compare_experiments(results_list)
        print(comparison_df.to_string(index=False))
    
    print(f"\\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()