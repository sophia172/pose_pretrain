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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_dataloaders, get_dataset_stats
from models.pretrain_model import PretrainModel
from models.loss import get_loss_fn
from trainers.trainer import Trainer, get_optimizer, get_scheduler


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
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
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


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration."""
    model_config = config.get("model", {})
    
    # Get model parameters
    num_joints = model_config.get("num_joints", 133)
    input_dim = model_config.get("input_dim", 2)
    output_dim = model_config.get("output_dim", 3)
    latent_dim = model_config.get("latent_dim", 256)
    
    # Architecture details
    architecture = model_config.get("architecture", {})
    encoder_hidden_dims = architecture.get("encoder_hidden_dims", [1024, 512])
    decoder_hidden_dims = architecture.get("decoder_hidden_dims", [512, 1024])
    num_encoder_layers = architecture.get("num_encoder_layers", 4)
    num_decoder_layers = architecture.get("num_decoder_layers", 4)
    num_heads = architecture.get("num_heads", 8)
    dropout = architecture.get("dropout", 0.1)
    activation = architecture.get("activation", "gelu")
    
    # Additional features
    use_positional_encoding = architecture.get("positional_encoding", True)
    use_joint_relations = architecture.get("use_joint_relations", True)
    
    # Loss weights
    consistency_loss_weight = model_config.get("consistency_loss_weight", 0.5)
    smoothness_loss_weight = model_config.get("smoothness_loss_weight", 0.1)
    
    # Create model
    model = PretrainModel(
        num_joints=num_joints,
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
        use_positional_encoding=use_positional_encoding,
        use_joint_relations=use_joint_relations,
        consistency_loss_weight=consistency_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight
    )
    
    return model


def main():
    """Main function for training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.debug:
        config["experiment"]["debug"] = True
    
    # Set up experiment name
    if args.experiment:
        experiment_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config['experiment']['name']}_{timestamp}"
    
    # Set up output directory
    output_dir = os.path.join(config["paths"]["output_dir"], experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir, debug=config["experiment"].get("debug", False))
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Set GPU device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Get dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader = get_dataloaders(config)
    
    # Create model
    logger.info("Creating model")
    model = create_model(config)
    
    # Create optimizer
    optimizer = get_optimizer(model, config)
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Create loss function
    loss_fn = get_loss_fn(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_epochs=config["training"].get("max_epochs", 100),
        clip_grad_norm=config["training"].get("clip_grad_norm"),
        save_every=config["training"].get("save_every", 10),
        log_every=config["training"].get("log_every", 100),
        val_every=config["training"].get("val_every", 1),
        save_best=config["training"].get("save_best", True),
        early_stopping_patience=config["training"].get("early_stopping_patience"),
        resume_from=args.resume,
        mixed_precision=config["training"].get("mixed_precision", False)
    )
    
    # Train model
    logger.info("Starting training")
    start_time = time.time()
    metrics = trainer.train()
    
    # Log training summary
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s ({training_time/60:.2f}m)")
    
    # Save final metrics
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main() 