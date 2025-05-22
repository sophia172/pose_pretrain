#!/usr/bin/env python
"""
Evaluation script for Human Pose Estimation models.

This script loads a trained model and evaluates its performance
using various metrics on the test dataset.
"""
from datetime import datetime
import os
import sys
import argparse
import logging
from colorama import Fore
import yaml
import json
import time
from typing import Dict, List, Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_dataloaders
from models.pretrain_model import PretrainModel
from models.loss import get_loss_fn
from trainers.trainer import Trainer
from utils.logger import get_logger, setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Human Pose Estimation model")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/evaluation",
                        help="Path to output directory")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results")
    parser.add_argument("--per_joint", action="store_true",
                        help="Compute per-joint metrics")
    parser.add_argument("--procrustes", action="store_true",
                        help="Apply Procrustes alignment before evaluation")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: str) -> PretrainModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if "model_config" in checkpoint:
        # If model config is saved in checkpoint
        model_config = checkpoint["model_config"]
        model = PretrainModel(**model_config)
    else:
        # Create model with default parameters
        model = PretrainModel()
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def procrustes_alignment(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Apply Procrustes alignment to bring the predicted pose into alignment with the target.
    
    Args:
        predicted: Predicted 3D pose of shape (num_joints, 3)
        target: Target 3D pose of shape (num_joints, 3)
        
    Returns:
        Aligned predicted pose of shape (num_joints, 3)
    """
    # Convert to numpy for SVD
    pred_np = predicted.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Centralize both poses
    pred_mean = np.mean(pred_np, axis=0)
    target_mean = np.mean(target_np, axis=0)
    
    pred_centered = pred_np - pred_mean
    target_centered = target_np - target_mean
    
    # Calculate optimal rotation
    H = pred_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate scaling factor
    scale = np.trace(target_centered @ target_centered.T) / np.trace(pred_centered @ pred_centered.T)
    scale = np.sqrt(scale)
    
    # Apply transformation
    aligned = scale * (pred_centered @ R) + target_mean
    
    return torch.from_numpy(aligned).to(predicted.device)


def calculate_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    apply_procrustes: bool = False,
    per_joint: bool = False
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted 3D poses of shape (batch_size, num_joints, 3)
        targets: Target 3D poses of shape (batch_size, num_joints, 3)
        apply_procrustes: Whether to apply Procrustes alignment before computing metrics
        per_joint: Whether to compute per-joint metrics
        
    Returns:
        Dictionary of metrics
    """
    batch_size, num_joints, _ = predictions.shape
    metrics = {}
    
    # Apply Procrustes alignment if requested
    if apply_procrustes:
        aligned_predictions = torch.zeros_like(predictions)
        for i in range(batch_size):
            aligned_predictions[i] = procrustes_alignment(predictions[i], targets[i])
        predictions = aligned_predictions
    
    # Calculate per-frame Mean Per Joint Position Error (MPJPE)
    joint_errors = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))  # (batch_size, num_joints)
    mpjpe_per_frame = torch.mean(joint_errors, dim=-1)  # (batch_size,)
    
    # Overall MPJPE
    mpjpe = torch.mean(joint_errors)
    metrics["mpjpe"] = mpjpe.item()
    
    # PA-MPJPE (Procrustes-aligned MPJPE) if alignment was applied
    if apply_procrustes:
        metrics["pa_mpjpe"] = mpjpe.item()
    
    # Calculate per-joint metrics if requested
    if per_joint:
        joint_mpjpe = torch.mean(joint_errors, dim=0)  # (num_joints,)
        metrics["per_joint_mpjpe"] = joint_mpjpe.cpu().numpy().tolist()
    
    # Calculate percentile errors
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    mpjpe_per_frame_np = mpjpe_per_frame.cpu().numpy()
    for p in percentiles:
        metrics[f"mpjpe_p{p}"] = float(np.percentile(mpjpe_per_frame_np, p))
    
    # Calculate Accuracy Under Curve metrics
    # AUC is calculated as percentage of frames with MPJPE below threshold
    thresholds = [30, 40, 50, 60, 70, 80, 90, 100, 150]  # mm
    for t in thresholds:
        accuracy = (mpjpe_per_frame < t).float().mean().item() * 100
        metrics[f"auc_{t}mm"] = accuracy
    
    # Calculate mean AUC across thresholds
    metrics["mean_auc"] = np.mean([metrics[f"auc_{t}mm"] for t in thresholds])
    
    return metrics


def visualize_results(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: Dict[str, Any],
    output_dir: str,
    limit_samples: int = 5
):
    """
    Visualize evaluation results.
    
    Args:
        predictions: Predicted 3D poses of shape (batch_size, num_joints, 3)
        targets: Target 3D poses of shape (batch_size, num_joints, 3)
        metrics: Dictionary of computed metrics
        output_dir: Directory to save visualizations
        limit_samples: Maximum number of samples to visualize
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot error distribution
    joint_errors = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))
    mpjpe_per_frame = torch.mean(joint_errors, dim=-1).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(mpjpe_per_frame, bins=50, alpha=0.7)
    plt.axvline(metrics["mpjpe"], color='r', linestyle='--', label=f'Mean: {metrics["mpjpe"]:.2f}')
    plt.axvline(metrics["mpjpe_p50"], color='g', linestyle='--', label=f'Median: {metrics["mpjpe_p50"]:.2f}')
    plt.title("MPJPE Distribution")
    plt.xlabel("MPJPE (mm)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(vis_dir, "mpjpe_distribution.png"), dpi=200)
    plt.close()
    
    # Plot per-joint errors if available
    if "per_joint_mpjpe" in metrics:
        plt.figure(figsize=(12, 6))
        joint_indices = range(len(metrics["per_joint_mpjpe"]))
        plt.bar(joint_indices, metrics["per_joint_mpjpe"])
        plt.axhline(metrics["mpjpe"], color='r', linestyle='--', label=f'Mean: {metrics["mpjpe"]:.2f}')
        plt.title("Per-Joint MPJPE")
        plt.xlabel("Joint Index")
        plt.ylabel("MPJPE (mm)")
        plt.xticks(joint_indices)
        plt.legend()
        plt.savefig(os.path.join(vis_dir, "per_joint_mpjpe.png"), dpi=200)
        plt.close()
    
    # Plot AUC curve
    thresholds = [30, 40, 50, 60, 70, 80, 90, 100, 150]
    auc_values = [metrics[f"auc_{t}mm"] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, auc_values, 'o-')
    plt.title("Accuracy Under Curve (AUC)")
    plt.xlabel("Threshold (mm)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, "auc_curve.png"), dpi=200)
    plt.close()
    
    # Visualize worst and best predictions
    errors = mpjpe_per_frame
    best_indices = np.argsort(errors)[:limit_samples]
    worst_indices = np.argsort(errors)[-limit_samples:]
    
    # Helper function to visualize a single pose comparison
    def visualize_pose_comparison(pred, target, output_path, title):
        fig = plt.figure(figsize=(12, 6))
        
        # Target pose
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Ground Truth")
        target_np = target.cpu().numpy()
        
        # Predicted pose
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Prediction")
        pred_np = pred.cpu().numpy()
        
        # Plot both poses
        for ax, pose in [(ax1, target_np), (ax2, pred_np)]:
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='b', s=20)
            
            # Draw connections between joints (if a skeleton is defined)
            # This should be adapted to the specific skeleton structure
            limb_connections = [
                (0, 1), (1, 2), (2, 3),  # Head, neck, spine
                (1, 4), (4, 5), (5, 6),  # Right arm
                (1, 7), (7, 8), (8, 9),  # Left arm
                (0, 10), (10, 11), (11, 12),  # Right leg
                (0, 13), (13, 14), (14, 15)   # Left leg
            ]
            
            for joint1, joint2 in limb_connections:
                if joint1 < len(pose) and joint2 < len(pose):
                    ax.plot(
                        [pose[joint1, 0], pose[joint2, 0]],
                        [pose[joint1, 1], pose[joint2, 1]],
                        [pose[joint1, 2], pose[joint2, 2]],
                        'r-'
                    )
            
            # Set equal aspect ratio
            max_range = np.array([
                pose[:, 0].max() - pose[:, 0].min(),
                pose[:, 1].max() - pose[:, 1].min(),
                pose[:, 2].max() - pose[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (pose[:, 0].max() + pose[:, 0].min()) * 0.5
            mid_y = (pose[:, 1].max() + pose[:, 1].min()) * 0.5
            mid_z = (pose[:, 2].max() + pose[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Add overall title
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
    
    # Visualize best predictions
    for i, idx in enumerate(best_indices):
        output_path = os.path.join(vis_dir, f"best_{i+1}.png")
        title = f"Best Prediction #{i+1} (MPJPE: {errors[idx]:.2f}mm)"
        visualize_pose_comparison(predictions[idx], targets[idx], output_path, title)
    
    # Visualize worst predictions
    for i, idx in enumerate(worst_indices):
        output_path = os.path.join(vis_dir, f"worst_{i+1}.png")
        title = f"Worst Prediction #{i+1} (MPJPE: {errors[idx]:.2f}mm)"
        visualize_pose_comparison(predictions[idx], targets[idx], output_path, title)


def evaluate_model(
    model: PretrainModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    apply_procrustes: bool = False,
    per_joint: bool = False
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    Evaluate the model on the provided dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing test data
        device: Device to use for evaluation
        apply_procrustes: Whether to apply Procrustes alignment before computing metrics
        per_joint: Whether to compute per-joint metrics
        
    Returns:
        Tuple of (metrics, predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    logger.info("Starting evaluation")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Extract data from batch
            if isinstance(batch, dict):
                keypoints_2d = batch["keypoints_2d"].to(device)
                keypoints_3d = batch["keypoints_3d"].to(device)
            else:
                keypoints_2d, keypoints_3d = batch
                keypoints_2d = keypoints_2d.to(device)
                keypoints_3d = keypoints_3d.to(device)
            
            # Run inference
            predictions = model.predict(keypoints_2d)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(keypoints_3d.cpu())
    
    # Combine results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_predictions, 
        all_targets,
        apply_procrustes=apply_procrustes,
        per_joint=per_joint
    )
    
    return metrics, all_predictions, all_targets


def main():
    """Main function for evaluation script."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Set up experiment name
    if args.experiment:
        experiment_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config['experiment']['name']}_{timestamp}"
    
    print(f"{Fore.GREEN}✓ Experiment name: {experiment_name}")
    
    # Set up output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"{Fore.GREEN}✓ Output directory created: {output_dir}")
    
    # Set up logging
    setup_logging(output_dir, debug=config["experiment"].get("debug", 0))
    logger = get_logger(__name__)
    
    
    # Save a copy of the config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Override batch size in config
    config_copy = config.copy()
    config_copy["data"]["batch_size"] = args.batch_size
    
    # Get test dataloader
    logger.info("Creating test dataloader")
    _, test_loader = get_dataloaders(config_copy)
    
    if test_loader is None:
        logger.error("No test dataloader available")
        return
    
    # Evaluate model
    start_time = time.time()
    metrics, predictions, targets = evaluate_model(
        model, 
        test_loader, 
        device,
        apply_procrustes=args.procrustes,
        per_joint=args.per_joint
    )
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        elif name == "per_joint_mpjpe":
            logger.info(f"  {name}: [min: {min(value):.4f}, max: {max(value):.4f}]")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # Visualize results if requested
    if args.visualize:
        logger.info("Generating visualizations")
        visualize_results(predictions, targets, metrics, output_dir)
        logger.info(f"Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main() 