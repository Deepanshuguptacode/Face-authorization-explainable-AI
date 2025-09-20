"""
Training Framework for Face Recognition
=======================================

This module provides:
- Training loop with mixed precision
- Checkpointing and model saving
- Logging with TensorBoard and Weights & Biases
- Validation and metrics computation
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.face_recognition_model import FaceRecognitionModel, create_model
from models.losses import MultiTaskLoss, create_loss_function
from data.dataset import create_dataloaders


class Trainer:
    """
    Face Recognition Model Trainer with multi-task learning support
    """
    
    def __init__(self,
                 model: FaceRecognitionModel,
                 loss_fn: MultiTaskLoss,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_amp: bool = True,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 use_wandb: bool = False,
                 wandb_project: str = 'face-recognition',
                 experiment_name: str = 'baseline'):
        """
        Args:
            model: Face recognition model
            loss_fn: Multi-task loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            use_amp: Use automatic mixed precision
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            use_wandb: Use Weights & Biases logging
            wandb_project: W&B project name
            experiment_name: Experiment name for logging
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize AMP scaler
        if use_amp:
            self.scaler = GradScaler()
        
        # Initialize logging
        self.writer = SummaryWriter(os.path.join(log_dir, experiment_name))
        
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config={
                    'model': model.__class__.__name__,
                    'optimizer': optimizer.__class__.__name__,
                    'scheduler': scheduler.__class__.__name__ if scheduler else None,
                    'device': str(device),
                    'use_amp': use_amp
                }
            )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_identity_acc': [],
            'val_identity_acc': [],
            'train_attribute_acc': [],
            'val_attribute_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, 
                   train_loader,
                   train_identity: bool = True,
                   train_attributes: bool = True) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            train_identity: Whether to train identity head
            train_attributes: Whether to train attribute heads
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Metrics
        total_loss = 0.0
        total_identity_loss = 0.0
        total_attribute_loss = 0.0
        total_identity_correct = 0
        total_attribute_correct = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            
            identity_labels = None
            attribute_labels = None
            
            if train_identity and 'identity' in batch:
                identity_labels = batch['identity'].to(self.device)
            
            if train_attributes and 'attributes' in batch:
                attribute_labels = batch['attributes'].to(self.device)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    # Model forward
                    outputs = self.model(images, identity_labels)
                    
                    # Compute losses
                    losses = self.loss_fn(
                        identity_logits=outputs['identity_logits'],
                        attribute_logits=outputs['attribute_logits'],
                        identity_labels=identity_labels,
                        attribute_labels=attribute_labels,
                        train_identity=train_identity,
                        train_attributes=train_attributes
                    )
                    
                    loss = losses['total_loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Model forward
                outputs = self.model(images, identity_labels)
                
                # Compute losses
                losses = self.loss_fn(
                    identity_logits=outputs['identity_logits'],
                    attribute_logits=outputs['attribute_logits'],
                    identity_labels=identity_labels,
                    attribute_labels=attribute_labels,
                    train_identity=train_identity,
                    train_attributes=train_attributes
                )
                
                loss = losses['total_loss']
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            
            if 'identity_loss' in losses:
                total_identity_loss += losses['identity_loss'].item() * batch_size
                
                # Identity accuracy
                if train_identity and identity_labels is not None:
                    _, predicted = torch.max(outputs['identity_logits'], 1)
                    total_identity_correct += (predicted == identity_labels).sum().item()
            
            if 'attribute_loss' in losses:
                total_attribute_loss += losses['attribute_loss'].item() * batch_size
                
                # Attribute accuracy (binary)
                if train_attributes and attribute_labels is not None:
                    predicted_attrs = torch.sign(outputs['attribute_logits'])
                    total_attribute_correct += (predicted_attrs == attribute_labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log batch metrics
            self.global_step += 1
            
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                
                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_identity_loss': total_identity_loss / total_samples if train_identity else 0.0,
            'train_attribute_loss': total_attribute_loss / total_samples if train_attributes else 0.0,
        }
        
        if train_identity and identity_labels is not None:
            metrics['train_identity_acc'] = total_identity_correct / total_samples
        
        if train_attributes and attribute_labels is not None:
            num_attr_predictions = total_samples * attribute_labels.size(1)
            metrics['train_attribute_acc'] = total_attribute_correct / num_attr_predictions
        
        return metrics
    
    def validate(self, 
                val_loader,
                validate_identity: bool = True,
                validate_attributes: bool = True) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            validate_identity: Whether to validate identity
            validate_attributes: Whether to validate attributes
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Metrics
        total_loss = 0.0
        total_identity_loss = 0.0
        total_attribute_loss = 0.0
        total_identity_correct = 0
        total_attribute_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device)
                
                identity_labels = None
                attribute_labels = None
                
                if validate_identity and 'identity' in batch:
                    identity_labels = batch['identity'].to(self.device)
                
                if validate_attributes and 'attributes' in batch:
                    attribute_labels = batch['attributes'].to(self.device)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                losses = self.loss_fn(
                    identity_logits=outputs['identity_logits'],
                    attribute_logits=outputs['attribute_logits'],
                    identity_labels=identity_labels,
                    attribute_labels=attribute_labels,
                    train_identity=validate_identity,
                    train_attributes=validate_attributes
                )
                
                loss = losses['total_loss']
                
                # Update metrics
                total_loss += loss.item() * batch_size
                
                if 'identity_loss' in losses:
                    total_identity_loss += losses['identity_loss'].item() * batch_size
                    
                    # Identity accuracy
                    if validate_identity and identity_labels is not None:
                        _, predicted = torch.max(outputs['identity_logits'], 1)
                        total_identity_correct += (predicted == identity_labels).sum().item()
                
                if 'attribute_loss' in losses:
                    total_attribute_loss += losses['attribute_loss'].item() * batch_size
                    
                    # Attribute accuracy
                    if validate_attributes and attribute_labels is not None:
                        predicted_attrs = torch.sign(outputs['attribute_logits'])
                        total_attribute_correct += (predicted_attrs == attribute_labels).sum().item()
        
        # Compute metrics
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_identity_loss': total_identity_loss / total_samples if validate_identity else 0.0,
            'val_attribute_loss': total_attribute_loss / total_samples if validate_attributes else 0.0,
        }
        
        if validate_identity and identity_labels is not None:
            metrics['val_identity_acc'] = total_identity_correct / total_samples
        
        if validate_attributes and attribute_labels is not None:
            num_attr_predictions = total_samples * attribute_labels.size(1)
            metrics['val_attribute_acc'] = total_attribute_correct / num_attr_predictions
        
        return metrics
    
    def save_checkpoint(self, 
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       save_optimizer: bool = True) -> str:
        """
        Save model checkpoint
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            save_optimizer: Whether to save optimizer state
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.use_amp:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'{self.experiment_name}_epoch_{self.epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                f'{self.experiment_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(
            self.checkpoint_dir,
            f'{self.experiment_name}_latest.pth'
        )
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', {})
        
        # Load optimizer
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
        return checkpoint
    
    def train(self,
              train_loader,
              val_loader,
              num_epochs: int,
              train_identity: bool = True,
              train_attributes: bool = True,
              validate_every: int = 1,
              save_every: int = 5,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            train_identity: Whether to train identity head
            train_attributes: Whether to train attribute heads
            validate_every: Validation frequency (epochs)
            save_every: Checkpoint saving frequency (epochs)
            early_stopping_patience: Early stopping patience
        
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Training identity: {train_identity}")
        print(f"Training attributes: {train_attributes}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(
                train_loader, 
                train_identity=train_identity,
                train_attributes=train_attributes
            )
            
            # Update history
            for key, value in train_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)
            
            # Validation
            if epoch % validate_every == 0:
                val_metrics = self.validate(
                    val_loader,
                    validate_identity=train_identity,
                    validate_attributes=train_attributes
                )
                
                # Update history
                for key, value in val_metrics.items():
                    if key not in self.training_history:
                        self.training_history[key] = []
                    self.training_history[key].append(value)
                
                # Check for best model
                is_best = False
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Log metrics
                self.log_metrics(train_metrics, val_metrics, epoch)
                
                # Early stopping
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    all_metrics = {**train_metrics, **val_metrics}
                    self.save_checkpoint(all_metrics, is_best=is_best)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'] if 'val_loss' in locals() else train_metrics['train_loss'])
                else:
                    self.scheduler.step()
        
        # Training completed
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save final checkpoint
        final_metrics = {**train_metrics}
        if 'val_metrics' in locals():
            final_metrics.update(val_metrics)
        
        self.save_checkpoint(final_metrics)
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return self.training_history
    
    def log_metrics(self, 
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float],
                   epoch: int):
        """Log metrics to TensorBoard and W&B"""
        
        # TensorBoard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # W&B logging
        if self.use_wandb:
            log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
        
        # Console logging
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_metrics.get('train_loss', 0):.4f}")
        print(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        
        if 'train_identity_acc' in train_metrics:
            print(f"  Train Identity Acc: {train_metrics['train_identity_acc']:.4f}")
        if 'val_identity_acc' in val_metrics:
            print(f"  Val Identity Acc: {val_metrics['val_identity_acc']:.4f}")
        
        if 'train_attribute_acc' in train_metrics:
            print(f"  Train Attribute Acc: {train_metrics['train_attribute_acc']:.4f}")
        if 'val_attribute_acc' in val_metrics:
            print(f"  Val Attribute Acc: {val_metrics['val_attribute_acc']:.4f}")


def create_trainer(model_config: Dict,
                  loss_config: Dict,
                  data_config: Dict,
                  train_config: Dict) -> Tuple[Trainer, Any, Any, Any]:
    """
    Factory function to create trainer with all components
    
    Args:
        model_config: Model configuration
        loss_config: Loss configuration  
        data_config: Data configuration
        train_config: Training configuration
    
    Returns:
        Tuple of (trainer, train_loader, val_loader, test_loader)
    """
    # Create data loaders
    train_loader, val_loader, test_loader, identity_encoding = create_dataloaders(data_config)
    
    # Update model config with actual number of classes
    model_config['num_classes'] = len(identity_encoding)
    
    # Create model
    model = create_model(model_config)
    
    # Create loss function with attribute weights
    if train_loader.dataset.return_attributes:
        attribute_weights = train_loader.dataset.get_attribute_weights()
        loss_fn = create_loss_function(loss_config, attribute_weights)
    else:
        loss_fn = create_loss_function(loss_config)
    
    # Create optimizer
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    lr = train_config.get('learning_rate', 1e-3)
    weight_decay = train_config.get('weight_decay', 1e-4)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = train_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create scheduler
    scheduler = None
    scheduler_config = train_config.get('scheduler', None)
    if scheduler_config:
        scheduler_name = scheduler_config.get('name', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            T_max = scheduler_config.get('T_max', train_config.get('num_epochs', 100))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'plateau':
            patience = scheduler_config.get('patience', 5)
            factor = scheduler_config.get('factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor
            )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        use_amp=train_config.get('use_amp', True),
        checkpoint_dir=train_config.get('checkpoint_dir', 'checkpoints'),
        log_dir=train_config.get('log_dir', 'logs'),
        use_wandb=train_config.get('use_wandb', False),
        wandb_project=train_config.get('wandb_project', 'face-recognition'),
        experiment_name=train_config.get('experiment_name', 'baseline')
    )
    
    return trainer, train_loader, val_loader, test_loader


# Training configuration presets
TRAIN_CONFIGS = {
    'baseline': {
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'scheduler': {
            'name': 'cosine',
            'T_max': 50
        },
        'use_amp': True,
        'device': 'cuda',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'use_wandb': False,
        'experiment_name': 'baseline'
    },
    'fast_test': {
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 5,
        'use_amp': True,
        'device': 'cuda',
        'experiment_name': 'fast_test'
    }
}


if __name__ == "__main__":
    # Test trainer creation
    from models.face_recognition_model import MODEL_CONFIGS
    from models.losses import LOSS_CONFIGS
    from data.dataset import DATA_CONFIGS
    
    print("Testing trainer creation...")
    
    try:
        trainer, train_loader, val_loader, test_loader = create_trainer(
            model_config=MODEL_CONFIGS['baseline'],
            loss_config=LOSS_CONFIGS['baseline'],
            data_config=DATA_CONFIGS['baseline'],
            train_config=TRAIN_CONFIGS['fast_test']
        )
        
        print("✓ Trainer created successfully")
        print(f"  - Model: {trainer.model.__class__.__name__}")
        print(f"  - Device: {trainer.device}")
        print(f"  - Data loaders: {len(train_loader)}/{len(val_loader)}/{len(test_loader)} batches")
        
        # Test a training step
        print("\\nTesting training step...")
        batch = next(iter(train_loader))
        trainer.model.train()
        
        images = batch['image'].to(trainer.device)
        outputs = trainer.model(images)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {images.shape}")
        print(f"  - Identity logits: {outputs['identity_logits'].shape}")
        print(f"  - Attribute logits: {outputs['attribute_logits'].shape}")
        
        print("\\n✓ Trainer test passed!")
        
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()