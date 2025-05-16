#!/usr/bin/env python
"""
Main training script for Human Pose Estimation models.

This script loads configurations, datasets, and models to train
a pose estimation model.
"""
import os
import sys
import argparse
import logging
import yaml
import time
from datetime import datetime
from typing import Dict, Any
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Progress bar and colorful console output
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

# Add src directory to path for both running from project root and from within src
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
project_root = os.path.abspath(os.path.join(src_dir, ".."))

# Ensure project root is in the Python path for absolute imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataloader import get_dataloaders, get_dataset_stats
from src.models.pretrain_model import PretrainModel
from src.models.pretrain_vit_model import PretrainViTModel
from src.models.loss import get_loss_fn
from src.trainers.trainer import Trainer, get_optimizer, get_scheduler
from src.utils.device import get_device, print_device_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Human Pose Estimation model")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (default: timestamp)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--model_type", type=str, default="transformer",
                        help="Model type to use (transformer or vit)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    print(f"{Fore.CYAN}⚙️  Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str, debug: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file handler
    log_file = os.path.join(log_dir, "train.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_model(config: Dict[str, Any], model_type: str = "transformer") -> nn.Module:
    """Create model based on configuration and model type."""
    model_config = config.get("model", {})
    model_config = load_config(model_config.get("config_file", "config/models/pretrain_model1.yaml"))
    architecture = model_config.get("architecture", {})
    
    # Get model parameters
    num_joints = config.get("num_joints", 133)
    input_dim = config.get("input_dim", 2)
    output_dim = config.get("output_dim", 3)
    
    print(f"{Fore.GREEN}🔨 Building {model_type.upper()} model with:")
    print(f"{Fore.GREEN}   - {num_joints} joints")
    print(f"{Fore.GREEN}   - {input_dim}D input → {output_dim}D output")
    
    # Create model based on specified type
    if model_type.lower() == "vit":
        # Vision Transformer model
        vit_config = model_config.get("vit", {})
        
        embed_dim = vit_config.get("embed_dim", 256)
        latent_dim = model_config.get("latent_dim", 256)
        encoder_depth = vit_config.get("encoder_depth", 6)
        decoder_depth = vit_config.get("decoder_depth", 6)
        num_heads = vit_config.get("num_heads", 8)
        mlp_ratio = vit_config.get("mlp_ratio", 4)
        dropout = vit_config.get("dropout", 0.1)
        attention_dropout = vit_config.get("attention_dropout", 0.1)
        activation = vit_config.get("activation", "gelu")
        use_positional_encoding = vit_config.get("use_positional_encoding", True)
        use_joint_relations = vit_config.get("use_joint_relations", True)
        
        # Loss weights
        loss_config = model_config.get("training", {}).get("losses", {})
        consistency_loss_weight = loss_config.get("consistency", {}).get("weight", 0.5)
        smoothness_loss_weight = loss_config.get("smoothness", {}).get("weight", 0.1)
        
        # Print ViT specifics
        print(f"{Fore.GREEN}   - ViT specs: {encoder_depth} encoder layers, {decoder_depth} decoder layers, {num_heads} heads")
        
        model = PretrainViTModel(
            num_joints=num_joints,
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations,
            consistency_loss_weight=consistency_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight
        )
    else:
        # Standard transformer model
        latent_dim = architecture.get("latent", {}).get("dim", 256)
        
        encoder_config = architecture.get("encoder", {})
        decoder_config = architecture.get("decoder", {})
        encoder_hidden_dims = encoder_config.get("hidden_dims", [1024, 512])
        num_encoder_layers = encoder_config.get("num_layers", 4)
        num_encoder_heads = encoder_config.get("num_heads", 8)
        decoder_hidden_dims = decoder_config.get("hidden_dims", [1024, 512])
        num_decoder_layers = decoder_config.get("num_layers", 4)
        num_decoder_heads = decoder_config.get("num_heads", 4)
        
        dropout = decoder_config.get("dropout", 0.1)
        activation = decoder_config.get("activation", "gelu")
        
        # Additional features
        use_positional_encoding = architecture.get("positional_encoding", True)
        use_joint_relations = architecture.get("use_joint_relations", True)
        
        # Loss weights
        loss_config = model_config.get("training", {}).get("losses", {})
        consistency_loss_weight = loss_config.get("consistency", {}).get("weight", 0.5)
        smoothness_loss_weight = loss_config.get("smoothness", {}).get("weight", 0.1)
        
        # Print transformer specifics
        print(f"{Fore.GREEN}   - Transformer specs: {num_encoder_layers} encoder layers, {num_decoder_layers} decoder layers")
        print(f"{Fore.GREEN}   - Hidden dims: {encoder_hidden_dims} → {decoder_hidden_dims}")
        
        model = PretrainModel(
            num_joints=num_joints,
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_encoder_heads,
            dropout=dropout,
            activation=activation,
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations,
            consistency_loss_weight=consistency_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight
        )
    
    # Print model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{Fore.GREEN}   - Total parameters: {param_count:,}")
    
    return model


def print_progress_header(text, width=80):
    """Print a centered header with decoration."""
    padding = max(0, (width - len(text) - 4) // 2)
    print(f"\n{Fore.YELLOW}{'=' * width}")
    print(f"{Fore.YELLOW}={' ' * padding}{Fore.CYAN}{text}{Fore.YELLOW}{' ' * padding}=")
    print(f"{Fore.YELLOW}{'=' * width}\n")


class TrainingProgressCallback:
    """Callback for tracking training progress with a progress bar."""
    
    def __init__(self, max_epochs, verbose=False):
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.pbar = None
        self.verbose = verbose
        
    def on_epoch_start(self, epoch):
        """Called at the start of each epoch."""
        self.current_epoch = epoch
        if self.pbar is not None:
            self.pbar.close()
        
        # Create a new progress bar for this epoch
        desc = f"{Fore.BLUE}Epoch {epoch+1}/{self.max_epochs}"
        self.pbar = tqdm(total=100, desc=desc, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        
    def on_batch_end(self, batch_idx, num_batches, loss):
        """Called after each batch."""
        # Update progress bar
        progress = int(100 * batch_idx / num_batches)
        self.pbar.update(progress - self.pbar.n)
        
        if self.verbose and batch_idx % max(1, num_batches // 10) == 0:
            # Print more detailed progress every 10% of batches
            self.pbar.set_postfix(loss=f"{loss:.4f}")
        
    def on_epoch_end(self, train_loss, val_loss=None):
        """Called at the end of each epoch."""
        # Complete the progress bar
        self.pbar.update(100 - self.pbar.n)
        
        # Print summary for this epoch
        if val_loss is not None:
            message = f"{Fore.GREEN}Epoch {self.current_epoch+1}/{self.max_epochs} completed: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        else:
            message = f"{Fore.GREEN}Epoch {self.current_epoch+1}/{self.max_epochs} completed: train_loss={train_loss:.4f}"
        
        self.pbar.set_description(message)
        time.sleep(0.1)  # Allow tqdm to refresh
        
    def close(self):
        """Clean up resources."""
        if self.pbar is not None:
            self.pbar.close()


def update_progress(progress_value):
    """Update progress callback function for trainer."""
    # This function is a placeholder that will be replaced by TrainingProgressCallback
    pass


def main():
    """Main function for training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Print welcome message
    print_progress_header("Human Pose Estimation Pretraining")
    
    print(f"{Fore.CYAN}🔧 Initializing training environment...")
    
    # Set random seed
    set_seed(args.seed)
    print(f"{Fore.GREEN}✓ Random seed set to {args.seed}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.debug:
        config["experiment"]["debug"] = True
        print(f"{Fore.YELLOW}⚠️ Debug mode enabled")
    
    # Set up experiment name
    if args.experiment:
        experiment_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config['experiment']['name']}_{timestamp}"
    
    print(f"{Fore.GREEN}✓ Experiment name: {experiment_name}")
    
    # Set up output directory
    output_dir = os.path.join(config["paths"]["output_dir"], experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"{Fore.GREEN}✓ Output directory created: {output_dir}")
    
    # Set up logging
    setup_logging(output_dir, debug=config["experiment"].get("debug", False))
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Print progress header for device setup
    print_progress_header("Device Configuration")
    
    # Set GPU device
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")
    
    # Print detailed device information
    print_device_info()
    
    # Print progress header for data loading
    print_progress_header("Data Loading")
    
    # Get dataloaders
    print(f"{Fore.CYAN}📊 Loading datasets...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Print dataset info
    print(f"{Fore.GREEN}✓ Training set: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    if val_loader:
        print(f"{Fore.GREEN}✓ Validation set: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    else:
        print(f"{Fore.YELLOW}⚠️ No validation set provided")
    
    # Print progress header for model creation
    print_progress_header("Model Creation")
    
    # Determine model type to use
    model_type = args.model_type
    if "model_type" in config["model"]:
        model_type = config["model"]["model_type"]
    
    # Create model
    model = create_model(config, model_type)
    
    # Create optimizer
    print(f"{Fore.CYAN}🔧 Creating optimizer...")
    optimizer = get_optimizer(model, config)
    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam")
    lr = training_config.get("learning_rate", 0.0001)
    print(f"{Fore.GREEN}✓ Using {optimizer_type.upper()} optimizer with learning rate {lr}")
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, config)
    if scheduler:
        lr_scheduler_config = training_config.get("lr_scheduler", {})
        scheduler_type = lr_scheduler_config.get("type", "none")
        print(f"{Fore.GREEN}✓ Using {scheduler_type.upper()} learning rate scheduler")
    else:
        print(f"{Fore.YELLOW}ℹ️ No learning rate scheduler specified")
    
    # Create loss function
    loss_fn = get_loss_fn(config)
    print(f"{Fore.GREEN}✓ Using loss function: {loss_fn.description}")
    
    # Print progress header for training setup
    print_progress_header("Training Setup")
    
    # Training parameters
    epochs = training_config.get("epochs", 100)
    print(f"{Fore.CYAN}🔄 Training for {epochs} epochs")
    
    if args.resume:
        print(f"{Fore.CYAN}🔄 Resuming training from {args.resume}")
    
    # Create progress callback
    progress_callback = TrainingProgressCallback(epochs, verbose=args.verbose)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_epochs=epochs,
        clip_grad_norm=training_config.get("grad_clip"),
        save_every=config["experiment"].get("save_every", 10),
        log_every=config["experiment"].get("log_every", 100),
        val_every=config["experiment"].get("val_every", 1),
        save_best=config["experiment"].get("save_best", True),
        early_stopping_patience=training_config.get("early_stopping", {}).get("patience"),
        resume_from=args.resume,
        mixed_precision=training_config.get("mixed_precision", False),
        progress_fn=update_progress
    )
    
    # Print progress header for training
    print_progress_header("Training")
    
    # Add progress callback to trainer
    trainer.set_progress_callback(progress_callback.on_epoch_start, 
                                 progress_callback.on_batch_end,
                                 progress_callback.on_epoch_end)
    
    # Train model
    print(f"{Fore.CYAN}🚀 Starting training...")
    start_time = time.time()
    metrics = trainer.train()
    
    # Calculate training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Clean up progress callback
    progress_callback.close()
    
    # Print progress header for results
    print_progress_header("Training Complete")
    
    # Log final metrics
    print(f"{Fore.GREEN}✅ Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{Fore.GREEN}✅ Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"{Fore.GREEN}✅ Model saved to {output_dir}")
    
    # Create a simple ASCII chart of loss progression
    if len(metrics["train_loss"]) > 0:
        print(f"\n{Fore.CYAN}📈 Training Loss Progression:")
        max_len = 40  # Max chart width
        
        # Filter out NaN values before calculating min/max
        valid_losses = [loss for loss in metrics["train_loss"] if not torch.isnan(loss).any() if not math.isnan(loss) if loss is not None]
        
        if not valid_losses:
            print(f"{Fore.YELLOW}⚠️ Cannot visualize training loss - NaN values detected")
        else:
            max_loss = max(valid_losses)
            min_loss = min(valid_losses)
            loss_range = max_loss - min_loss
            
            for i, loss in enumerate(metrics["train_loss"]):
                if i % max(1, len(metrics["train_loss"]) // 10) == 0:  # Print ~10 points
                    if math.isnan(loss) or torch.isnan(torch.tensor(loss)).any():
                        print(f"{Fore.BLUE}Epoch {i+1:3d}: {Fore.RED}NaN    {Fore.YELLOW}{'!' * max_len}")
                    else:
                        normalized = int((loss - min_loss) / (loss_range + 1e-8) * max_len) if loss_range > 0 else 0
                        bar = '█' * (max_len - normalized) + '░' * normalized
                        print(f"{Fore.BLUE}Epoch {i+1:3d}: {loss:.4f} {Fore.YELLOW}{bar}")
    
    logger.info(f"Training completed with final metrics: {metrics}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    return 0

def cli():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Training interrupted by user. Exiting gracefully...")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 

if __name__ == "__main__":
    cli()