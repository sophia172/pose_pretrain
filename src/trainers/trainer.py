"""
Trainer module for Human Pose Estimation models.

This module provides a trainer class that handles model training,
evaluation, and logging.
"""
import os
import time
import logging
import platform
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


from src.utils.device import get_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for Human Pose Estimation models.
    
    Handles model training, evaluation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs",
        experiment_name: str = "pretrain",
        max_epochs: int = 100,
        clip_grad_norm: Optional[float] = None,
        save_every: int = 10,
        log_every: int = 100,
        val_every: int = 1,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None,
        mixed_precision: bool = False,
        progress_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Optional learning rate scheduler
            loss_fn: Loss function
            device: Device to use for training
            output_dir: Directory to save outputs
            experiment_name: Name of the experiment
            max_epochs: Maximum number of epochs to train for
            clip_grad_norm: Max norm for gradient clipping
            save_every: Save model every N epochs
            log_every: Log metrics every N iterations
            val_every: Validate model every N epochs
            save_best: Whether to save the best model
            early_stopping_patience: Patience for early stopping
            resume_from: Path to checkpoint to resume from
            mixed_precision: Whether to use mixed precision training
            progress_fn: Optional function to update progress display
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        self.max_epochs = max_epochs
        self.clip_grad_norm = clip_grad_norm
        self.save_every = save_every
        self.log_every = log_every
        self.val_every = val_every
        self.save_best = save_best
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        self.progress_fn = progress_fn
        
        # Progress callbacks
        self.on_epoch_start_callback = None
        self.on_batch_end_callback = None
        self.on_epoch_end_callback = None
        
        # Set up model and optimizer
        self.model = self.model.to(self.device)
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
        # Set up mixed precision training if enabled
        self.scaler = None
        if self.mixed_precision:
            # MPS doesn't support amp yet, so only enable for CUDA
            if self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                logger.warning("Mixed precision training is only supported on CUDA devices. Disabling.")
                self.mixed_precision = False
        
        # Set up output directory
        self.output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Checkpoints directory
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        
        # Metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.learning_rates = []
        
        # Resume training if checkpoint provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)
            
        # Log trainer setup
        logger.info(f"Trainer initialized with {self.device} device")
        logger.info(f"Training for {self.max_epochs} epochs")
        logger.info(f"Outputs will be saved to {self.output_dir}")
        logger.info(f"Model has {self._count_parameters():,} parameters")
        
    def _count_parameters(self) -> int:
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def set_progress_callback(self, 
                             on_epoch_start: Optional[Callable] = None, 
                             on_batch_end: Optional[Callable] = None,
                             on_epoch_end: Optional[Callable] = None):
        """
        Set callbacks for progress reporting.
        
        Args:
            on_epoch_start: Called at the start of each epoch with epoch number
            on_batch_end: Called after each batch with batch_idx, num_batches, loss
            on_epoch_end: Called at the end of each epoch with train_loss and val_loss
        """
        self.on_epoch_start_callback = on_epoch_start
        self.on_batch_end_callback = on_batch_end
        self.on_epoch_end_callback = on_epoch_end
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary of training metrics history
        """
        logger.info("Starting training")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.max_epochs):
            # Call epoch start callback
            if self.on_epoch_start_callback:
                self.on_epoch_start_callback(epoch)
                
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            self.train_loss_history.append(train_metrics["loss"])
            
            # Validation metrics (if available)
            val_metrics = None
            
            # Validation if needed
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                val_metrics = self._validate(epoch)
                self.val_loss_history.append(val_metrics["loss"])
                
                # Save best model
                if self.save_best and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"Epoch {epoch+1}: New best model saved with validation loss {self.best_val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    
                # Early stopping
                if (self.early_stopping_patience is not None and 
                    self.epochs_without_improvement >= self.early_stopping_patience):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # If no validation, track training loss for best model
                current_loss = train_metrics["loss"]
                if self.save_best and current_loss < self.best_val_loss:
                    self.best_val_loss = current_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Call epoch end callback
            if self.on_epoch_end_callback:
                val_loss = val_metrics["loss"] if val_metrics else None
                self.on_epoch_end_callback(train_metrics["loss"], val_loss)
                    
            # Save checkpoint periodically
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)
                
            # LR scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs validation loss
                    if self.val_loader is not None:
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step(train_metrics["loss"])
                else:
                    # Other schedulers just step
                    self.scheduler.step()
                
            # Store current learning rate
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        # Save final model
        self._save_checkpoint(self.max_epochs - 1, is_final=True)
        
        # Calculate total training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        
        # Return metrics history
        metrics_history = {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history if self.val_loader is not None else [],
            "learning_rates": self.learning_rates
        }
        
        return metrics_history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics for the epoch
        """
        self.model.train()
        
        # Track metrics
        total_loss = 0.0
        samples_count = 0
        
        # Set up progress tracking
        start_time = time.time()
        num_batches = len(self.train_loader)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loaded train loader")
        # Iterate over batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Batch {batch_idx} started")
                
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device) if "target" in batch else None
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.mixed_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, targets) if targets is not None else self.model(inputs)
                    
                    if self.loss_fn is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        # If no loss function is provided, use the model's computed loss
                        loss = outputs["total_loss"] if "total_loss" in outputs else outputs["loss"]
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(inputs, targets) if targets is not None else self.model(inputs)
                
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    # If no loss function is provided, use the model's computed loss
                    loss = outputs["total_loss"] if "total_loss" in outputs else outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if enabled
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            samples_count += batch_size
            total_loss += loss.item() * batch_size
            
            # Log batch metrics
            if batch_idx % self.log_every == 0:
                batch_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{self.max_epochs} [{batch_idx}/{len(self.train_loader)}] "
                           f"Loss: {loss.item():.4f} Time: {batch_time:.2f}s")
                start_time = time.time()
            
            # Call batch end callback
            if self.on_batch_end_callback:
                self.on_batch_end_callback(batch_idx, num_batches, loss.item())
                
            # Update progress if function provided
            if self.progress_fn is not None:
                progress = (epoch + batch_idx / len(self.train_loader)) / self.max_epochs
                self.progress_fn(progress)
                
            # Update global step
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / samples_count if samples_count > 0 else 0.0
        
        # Log epoch metrics
        logger.info(f"Epoch {epoch+1}/{self.max_epochs} Training - Loss: {avg_loss:.4f}")
        
        # Return metrics
        return {"loss": avg_loss}

    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Track metrics
        total_loss = 0.0
        samples_count = 0
        
        # Disable gradients for validation
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device) if "target" in batch else None
                
                # Forward pass
                outputs = self.model(inputs, targets) if targets is not None else self.model(inputs)
                
                # Calculate loss
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    # If no loss function is provided, use the model's computed loss
                    loss = outputs["total_loss"] if "total_loss" in outputs else outputs["loss"]
                
                # Update metrics
                batch_size = inputs.size(0)
                samples_count += batch_size
                total_loss += loss.item() * batch_size
        
        # Calculate validation metrics
        avg_loss = total_loss / samples_count if samples_count > 0 else 0.0
        
        # Log validation metrics
        logger.info(f"Epoch {epoch+1}/{self.max_epochs} Validation - Loss: {avg_loss:.4f}")
        
        # Return metrics
        return {"loss": avg_loss}

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        # Save regular checkpoint
        if not is_best and not is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")
            
        # Save best model
        if is_best:
            best_model_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.debug(f"Saved best model to {best_model_path}")
            
        # Save final model
        if is_final:
            final_model_path = os.path.join(self.output_dir, "final_model.pt")
            torch.save(checkpoint, final_model_path)
            logger.debug(f"Saved final model to {final_model_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Check if file exists
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        # Load checkpoint on CPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Move optimizer state to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        # Load scheduler state if present
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load training state
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        
        # Track metrics
        total_loss = 0.0
        samples_count = 0
        
        # Additional metrics
        total_pred_errors = 0.0
        
        # Disable gradients for evaluation
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device) if "target" in batch else None
                
                # Forward pass
                outputs = self.model(inputs, targets) if targets is not None else self.model(inputs)
                
                # Get predictions
                if isinstance(outputs, dict):
                    predictions = outputs.get("pred", outputs.get("pred_3d", None))
                else:
                    predictions = outputs
                
                # Calculate loss
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs, targets)
                elif "total_loss" in outputs:
                    loss = outputs["total_loss"]
                elif "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    # Fallback to mean squared error
                    loss = torch.nn.functional.mse_loss(predictions, targets)
                
                # Calculate prediction error
                if targets is not None and predictions is not None:
                    pred_error = torch.mean(torch.norm(predictions - targets, dim=-1))
                    total_pred_errors += pred_error.item() * inputs.size(0)
                
                # Update metrics
                batch_size = inputs.size(0)
                samples_count += batch_size
                total_loss += loss.item() * batch_size
        
        # Calculate test metrics
        avg_loss = total_loss / samples_count if samples_count > 0 else 0.0
        avg_pred_error = total_pred_errors / samples_count if samples_count > 0 else 0.0
        
        # Log test metrics
        logger.info(f"Test - Loss: {avg_loss:.4f}, Prediction Error: {avg_pred_error:.4f}")
        
        # Return metrics
        metrics = {
            "loss": avg_loss,
            "prediction_error": avg_pred_error
        }
        
        return metrics


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    optimizer_config = config.get("optimizer", {})
    
    optimizer_type = optimizer_config.get("type", "adam").lower()
    lr = optimizer_config.get("learning_rate", 1e-3)
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    
    if optimizer_type == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        nesterov = optimizer_config.get("nesterov", False)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                         weight_decay=weight_decay, nesterov=nesterov)
    elif optimizer_type == "adam":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        return optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), 
                          weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        return optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), 
                           weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        PyTorch learning rate scheduler or None
    """
    train_config = config.get("training", {})
    scheduler_config = train_config.get("lr_scheduler", {})
    
    if "type" not in scheduler_config:
        return None
        
    scheduler_type = scheduler_config["type"].lower()
    
    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 30)
        gamma = scheduler_config.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "multistep":
        milestones = scheduler_config.get("milestones", [30, 60, 90])
        gamma = scheduler_config.get("gamma", 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = scheduler_config.get("patience", 10)
        factor = scheduler_config.get("factor", 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                 patience=patience, factor=factor)
    elif scheduler_type == "cosine":
        epochs = train_config.get("epochs", 100)
        warmup_epochs = scheduler_config.get("warmup_epochs", 0)
        eta_min = scheduler_config.get("eta_min", 0)
        
        if warmup_epochs > 0:
            # Create warmup + cosine annealing scheduler
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=eta_min
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # Create basic cosine annealing scheduler
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy model, data, and trainer for testing
    dummy_model = nn.Linear(10, 10)
    dummy_data = torch.randn(100, 10)
    dummy_target = torch.randn(100, 10)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_target)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)
    
    trainer = Trainer(
        model=dummy_model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        max_epochs=5,
        save_every=1
    )
    
    trainer.train() 