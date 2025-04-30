"""
Trainer module for Human Pose Estimation models.

This module provides a trainer class that handles model training,
evaluation, and logging.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        
        # Set up model and optimizer
        self.model = self.model.to(self.device)
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
        # Set up mixed precision training if enabled
        self.scaler = None
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
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
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary of training metrics history
        """
        logger.info("Starting training")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.max_epochs):
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            self.train_loss_history.append(train_metrics["loss"])
            
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
        
        # Return training history
        return {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history,
            "learning_rates": self.learning_rates
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                # Dict-style batch
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Get inputs and targets
                inputs_2d = batch.get("keypoints_2d")
                targets_3d = batch.get("keypoints_3d")
            else:
                # Tuple-style batch
                inputs_2d, targets_3d = batch
                inputs_2d = inputs_2d.to(self.device)
                targets_3d = targets_3d.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs_2d, targets_3d)
                    
                    if self.loss_fn is not None:
                        loss, loss_components = self.loss_fn(outputs["pred_3d"], targets_3d)
                    else:
                        # Use model's loss if no external loss function
                        loss = outputs["total_loss"]
                        loss_components = {k: v for k, v in outputs.items() if "loss" in k}
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Clip gradients if needed
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Update parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(inputs_2d, targets_3d)
                
                if self.loss_fn is not None:
                    loss, loss_components = self.loss_fn(outputs["pred_3d"], targets_3d)
                else:
                    # Use model's loss if no external loss function
                    loss = outputs["total_loss"]
                    loss_components = {k: v for k, v in outputs.items() if "loss" in k}
                
                # Backward pass
                loss.backward()
                
                # Clip gradients if needed
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Update parameters
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Accumulate loss components
            for k, v in loss_components.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            
            # Log progress
            if batch_idx % self.log_every == 0:
                batch_time = time.time() - epoch_start_time
                examples_per_sec = (batch_idx + 1) * self.train_loader.batch_size / batch_time
                
                log_msg = (f"Epoch: {epoch+1}/{self.max_epochs} "
                          f"[{batch_idx+1}/{num_batches} ({100. * (batch_idx+1) / num_batches:.0f}%)] "
                          f"Loss: {loss.item():.4f} "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                          f"Examples/sec: {examples_per_sec:.1f}")
                
                logger.info(log_msg)
                
            # Update progress display if provided
            if self.progress_fn is not None:
                progress = {
                    "epoch": epoch + 1,
                    "max_epochs": self.max_epochs,
                    "batch": batch_idx + 1,
                    "num_batches": num_batches,
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "examples_per_sec": (batch_idx + 1) * self.train_loader.batch_size / (time.time() - epoch_start_time)
                }
                self.progress_fn(progress)
        
        # Calculate average metrics
        epoch_loss /= num_batches
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= num_batches
            
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Train loss: {epoch_loss:.4f}")
        
        epoch_metrics["loss"] = epoch_loss
        return epoch_metrics
    
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {"loss": float('inf')}
            
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {}
        num_batches = len(self.val_loader)
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    # Dict-style batch
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    
                    # Get inputs and targets
                    inputs_2d = batch.get("keypoints_2d")
                    targets_3d = batch.get("keypoints_3d")
                else:
                    # Tuple-style batch
                    inputs_2d, targets_3d = batch
                    inputs_2d = inputs_2d.to(self.device)
                    targets_3d = targets_3d.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs_2d, targets_3d)
                
                # Calculate loss
                if self.loss_fn is not None:
                    loss, loss_components = self.loss_fn(outputs["pred_3d"], targets_3d)
                else:
                    # Use model's loss if no external loss function
                    loss = outputs["total_loss"]
                    loss_components = {k: v for k, v in outputs.items() if "loss" in k}
                
                # Update metrics
                val_loss += loss.item()
                
                # Accumulate loss components
                for k, v in loss_components.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        
        # Calculate average metrics
        val_loss /= num_batches
        for k in val_metrics.keys():
            val_metrics[k] /= num_batches
            
        # Calculate validation time
        val_time = time.time() - val_start_time
        
        # Log validation summary
        logger.info(f"Epoch {epoch+1} validation completed in {val_time:.2f}s - Val loss: {val_loss:.4f}")
        
        val_metrics["loss"] = val_loss
        return val_metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "learning_rates": self.learning_rates
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
        # Save final model if needed
        if is_final:
            final_path = os.path.join(self.output_dir, "final_model.pt")
            torch.save(checkpoint, final_path)
            
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load scaler state if available
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore training state
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint.get("global_step", 0)
        
        # Restore history
        self.train_loss_history = checkpoint.get("train_loss_history", [])
        self.val_loss_history = checkpoint.get("val_loss_history", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        
        logger.info(f"Resuming training from epoch {self.start_epoch}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        
        test_loss = 0.0
        test_metrics = {}
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    # Dict-style batch
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    
                    # Get inputs and targets
                    inputs_2d = batch.get("keypoints_2d")
                    targets_3d = batch.get("keypoints_3d")
                else:
                    # Tuple-style batch
                    inputs_2d, targets_3d = batch
                    inputs_2d = inputs_2d.to(self.device)
                    if targets_3d is not None:
                        targets_3d = targets_3d.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs_2d, targets_3d)
                pred_3d = outputs["pred_3d"]
                
                # Calculate loss if targets are available
                if targets_3d is not None:
                    if self.loss_fn is not None:
                        loss, loss_components = self.loss_fn(pred_3d, targets_3d)
                    else:
                        # Use model's loss if no external loss function
                        loss = outputs["total_loss"]
                        loss_components = {k: v for k, v in outputs.items() if "loss" in k}
                    
                    # Update metrics
                    test_loss += loss.item()
                    
                    # Accumulate loss components
                    for k, v in loss_components.items():
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        test_metrics[k] = test_metrics.get(k, 0.0) + v
                
                # Save predictions and targets for further analysis
                predictions.append(pred_3d.cpu().numpy())
                if targets_3d is not None:
                    targets.append(targets_3d.cpu().numpy())
        
        # Calculate average metrics
        num_batches = len(test_loader)
        test_loss /= num_batches
        for k in test_metrics.keys():
            test_metrics[k] /= num_batches
            
        # Combine predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        if targets:
            targets = np.concatenate(targets, axis=0)
            
            # Calculate MPJPE (Mean Per Joint Position Error)
            joint_error = np.mean(np.sqrt(np.sum((predictions - targets) ** 2, axis=-1)))
            test_metrics["mpjpe"] = joint_error
        
        test_metrics["loss"] = test_loss
        logger.info(f"Evaluation completed - Test loss: {test_loss:.4f}")
        
        return test_metrics


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    optim_config = config.get("optimizer", {})
    optim_type = optim_config.get("type", "adam").lower()
    lr = optim_config.get("lr", 1e-3)
    weight_decay = optim_config.get("weight_decay", 0.0)
    
    if optim_type == "adam":
        betas = optim_config.get("betas", (0.9, 0.999))
        eps = optim_config.get("eps", 1e-8)
        return optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optim_type == "adamw":
        betas = optim_config.get("betas", (0.9, 0.999))
        eps = optim_config.get("eps", 1e-8)
        return optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optim_type == "sgd":
        momentum = optim_config.get("momentum", 0.9)
        nesterov = optim_config.get("nesterov", False)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        PyTorch learning rate scheduler
    """
    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", None)
    
    if scheduler_type is None:
        return None
        
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 30)
        gamma = scheduler_config.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "multistep":
        milestones = scheduler_config.get("milestones", [30, 60, 90])
        gamma = scheduler_config.get("gamma", 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", 100)
        eta_min = scheduler_config.get("eta_min", 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == "plateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        threshold = scheduler_config.get("threshold", 1e-4)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, threshold=threshold
        )
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