#!/usr/bin/env python
"""
Inference script for Human Pose Estimation models.

This script loads a trained model and performs inference on test data
or single inputs.
"""
import os
import sys
import argparse
import logging
import yaml
import json
import time
from typing import Dict, List, Any, Union, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_dataloaders
from models.pretrain_model import PretrainModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained Human Pose Estimation model")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input JSON file or directory")
    parser.add_argument("--output", type=str, default="outputs/results",
                        help="Path to output directory")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: str) -> PretrainModel:
    """Load model from checkpoint."""
    # Load checkpoint
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


def load_keypoints_from_json(json_path: str) -> Dict[str, torch.Tensor]:
    """Load 2D keypoints from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract 2D keypoints based on the JSON structure
    # This may need to be adapted based on the exact JSON format
    if isinstance(data, dict):
        # Assuming the first level keys are frame IDs
        frames = sorted(data.keys())
        keypoints_2d = []
        
        for frame in frames:
            frame_data = data[frame]
            if "2d_keypoints" in frame_data:
                # Direct 2D keypoints
                keypoints = torch.tensor(frame_data["2d_keypoints"], dtype=torch.float32)
            elif "keypoints" in frame_data and "2d" in frame_data["keypoints"]:
                # Nested keypoints
                keypoints = torch.tensor(frame_data["keypoints"]["2d"], dtype=torch.float32)
            else:
                # Try to extract by joint indices
                joints = []
                for joint_idx in sorted(frame_data.keys()):
                    if "x" in frame_data[joint_idx] and "y" in frame_data[joint_idx]:
                        joints.append([
                            frame_data[joint_idx]["x"],
                            frame_data[joint_idx]["y"]
                        ])
                keypoints = torch.tensor(joints, dtype=torch.float32)
                
            keypoints_2d.append(keypoints)
            
        # Stack all frames
        keypoints_2d = torch.stack(keypoints_2d)
        
        return {"keypoints_2d": keypoints_2d}
    
    # If the JSON is an array or has a different structure
    logger.error(f"Unsupported JSON format: {json_path}")
    return {"keypoints_2d": torch.tensor([])}


def visualize_pose(keypoints_2d: torch.Tensor, keypoints_3d: torch.Tensor, 
                  output_path: str, limb_connections: Optional[List[tuple]] = None):
    """
    Visualize 2D and 3D poses.
    
    Args:
        keypoints_2d: 2D keypoints of shape (num_joints, 2)
        keypoints_3d: 3D keypoints of shape (num_joints, 3)
        output_path: Path to save visualization
        limb_connections: List of tuples defining limb connections
    """
    if limb_connections is None:
        # Default limb connections for visualization
        limb_connections = [
            (0, 1), (1, 2), (2, 3),  # Head, neck, spine
            (1, 4), (4, 5), (5, 6),  # Right arm
            (1, 7), (7, 8), (8, 9),  # Left arm
            (0, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15)   # Left leg
        ]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 6))
    
    # 2D pose subplot
    ax1 = fig.add_subplot(121)
    ax1.set_title("2D Pose")
    
    # Plot 2D keypoints
    ax1.scatter(keypoints_2d[:, 0].cpu().numpy(), keypoints_2d[:, 1].cpu().numpy(), c='b', s=20)
    
    # Plot limb connections
    for joint1, joint2 in limb_connections:
        if joint1 < len(keypoints_2d) and joint2 < len(keypoints_2d):
            ax1.plot(
                [keypoints_2d[joint1, 0].item(), keypoints_2d[joint2, 0].item()],
                [keypoints_2d[joint1, 1].item(), keypoints_2d[joint2, 1].item()],
                'r-'
            )
    
    ax1.set_aspect('equal')
    ax1.invert_yaxis()  # Invert y-axis to match image coordinates
    
    # 3D pose subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("3D Pose")
    
    # Plot 3D keypoints
    keypoints_3d_np = keypoints_3d.cpu().numpy()
    ax2.scatter(keypoints_3d_np[:, 0], keypoints_3d_np[:, 1], keypoints_3d_np[:, 2], c='b', s=20)
    
    # Plot limb connections
    for joint1, joint2 in limb_connections:
        if joint1 < len(keypoints_3d) and joint2 < len(keypoints_3d):
            ax2.plot(
                [keypoints_3d_np[joint1, 0], keypoints_3d_np[joint2, 0]],
                [keypoints_3d_np[joint1, 1], keypoints_3d_np[joint2, 1]],
                [keypoints_3d_np[joint1, 2], keypoints_3d_np[joint2, 2]],
                'r-'
            )
    
    # Set 3D axes properties
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Equal aspect ratio
    max_range = np.array([
        keypoints_3d_np[:, 0].max() - keypoints_3d_np[:, 0].min(),
        keypoints_3d_np[:, 1].max() - keypoints_3d_np[:, 1].min(),
        keypoints_3d_np[:, 2].max() - keypoints_3d_np[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (keypoints_3d_np[:, 0].max() + keypoints_3d_np[:, 0].min()) * 0.5
    mid_y = (keypoints_3d_np[:, 1].max() + keypoints_3d_np[:, 1].min()) * 0.5
    mid_z = (keypoints_3d_np[:, 2].max() + keypoints_3d_np[:, 2].min()) * 0.5
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def process_single_input(model: PretrainModel, input_path: str, output_dir: str, 
                        visualize: bool, device: str):
    """
    Process a single input file.
    
    Args:
        model: Trained model
        input_path: Path to input JSON file
        output_dir: Directory to save outputs
        visualize: Whether to visualize predictions
        device: Device to use for inference
    """
    logger.info(f"Processing input: {input_path}")
    
    # Load input data
    data = load_keypoints_from_json(input_path)
    
    # Prepare input
    keypoints_2d = data["keypoints_2d"].to(device)
    
    # Run inference
    with torch.no_grad():
        if len(keypoints_2d.shape) == 3:  # (num_frames, num_joints, 2)
            # Process sequence
            results = []
            for frame_idx in range(len(keypoints_2d)):
                frame_keypoints = keypoints_2d[frame_idx:frame_idx+1]
                prediction = model.predict_3d_from_2d(frame_keypoints)
                results.append(prediction)
                
            # Combine results
            keypoints_3d = torch.cat(results, dim=0)
        else:
            # Process single frame
            keypoints_3d = model.predict_3d_from_2d(keypoints_2d.unsqueeze(0))
            keypoints_3d = keypoints_3d.squeeze(0)
    
    # Save results
    output_file = os.path.join(output_dir, os.path.basename(input_path).replace(".json", "_3d.json"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create output JSON
    output_data = {}
    keypoints_3d_np = keypoints_3d.cpu().numpy()
    
    if len(keypoints_3d.shape) == 3:  # (num_frames, num_joints, 3)
        # Save sequence
        for frame_idx in range(len(keypoints_3d)):
            output_data[f"frame_{frame_idx}"] = {
                "keypoints_3d": keypoints_3d_np[frame_idx].tolist()
            }
    else:
        # Save single frame
        output_data["keypoints_3d"] = keypoints_3d_np.tolist()
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Visualize if requested
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        if len(keypoints_3d.shape) == 3:  # (num_frames, num_joints, 3)
            # Visualize sequence
            for frame_idx in range(len(keypoints_3d)):
                vis_file = os.path.join(vis_dir, f"{os.path.basename(input_path).replace('.json', '')}_frame_{frame_idx}.png")
                visualize_pose(
                    keypoints_2d[frame_idx], 
                    keypoints_3d[frame_idx], 
                    vis_file
                )
        else:
            # Visualize single frame
            vis_file = os.path.join(vis_dir, f"{os.path.basename(input_path).replace('.json', '')}.png")
            visualize_pose(keypoints_2d, keypoints_3d, vis_file)
            
        logger.info(f"Visualizations saved to: {vis_dir}")


def process_test_data(model: PretrainModel, config: Dict[str, Any], output_dir: str, 
                     batch_size: int, visualize: bool, device: str):
    """
    Process test data from dataloaders.
    
    Args:
        model: Trained model
        config: Configuration
        output_dir: Directory to save outputs
        batch_size: Batch size for inference
        visualize: Whether to visualize predictions
        device: Device to use for inference
    """
    logger.info("Processing test data from dataloader")
    
    # Override batch size in config
    config_copy = config.copy()
    config_copy["data"]["batch_size"] = batch_size
    
    # Get test dataloader
    _, val_loader = get_dataloaders(config_copy)
    
    if val_loader is None:
        logger.error("No validation/test dataloader available")
        return
    
    # Run inference
    all_predictions = []
    all_inputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            logger.info(f"Processing batch {batch_idx+1}/{len(val_loader)}")
            
            # Extract data from batch
            if isinstance(batch, dict):
                keypoints_2d = batch["keypoints_2d"].to(device)
                if "keypoints_3d" in batch:
                    keypoints_3d_gt = batch["keypoints_3d"].to(device)
                else:
                    keypoints_3d_gt = None
            else:
                keypoints_2d, keypoints_3d_gt = batch
                keypoints_2d = keypoints_2d.to(device)
                if keypoints_3d_gt is not None:
                    keypoints_3d_gt = keypoints_3d_gt.to(device)
            
            # Run inference
            prediction = model.predict_3d_from_2d(keypoints_2d)
            
            # Store results
            all_predictions.append(prediction.cpu())
            all_inputs.append(keypoints_2d.cpu())
            if keypoints_3d_gt is not None:
                all_targets.append(keypoints_3d_gt.cpu())
    
    # Combine results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    
    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mpjpe = torch.mean(torch.sqrt(torch.sum((all_predictions - all_targets) ** 2, dim=-1)))
        logger.info(f"Mean Per Joint Position Error (MPJPE): {mpjpe.item():.4f}")
        
        # Save metrics
        metrics = {
            "mpjpe": float(mpjpe.item()),
            "num_samples": len(all_predictions)
        }
        
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to: {metrics_file}")
    
    # Save sample predictions
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    num_samples = min(10, len(all_predictions))
    for i in range(num_samples):
        sample_file = os.path.join(samples_dir, f"sample_{i}.json")
        
        sample_data = {
            "keypoints_2d": all_inputs[i].numpy().tolist(),
            "keypoints_3d_pred": all_predictions[i].numpy().tolist()
        }
        
        if all_targets:
            sample_data["keypoints_3d_gt"] = all_targets[i].numpy().tolist()
            
        with open(sample_file, "w") as f:
            json.dump(sample_data, f, indent=2)
            
        # Visualize if requested
        if visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_file = os.path.join(vis_dir, f"sample_{i}.png")
            visualize_pose(all_inputs[i], all_predictions[i], vis_file)
    
    logger.info(f"Sample predictions saved to: {samples_dir}")
    if visualize:
        logger.info(f"Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")


def main():
    """Main function for inference script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    global logger
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Process input
    if args.input:
        if os.path.isfile(args.input):
            # Process single file
            process_single_input(model, args.input, output_dir, args.visualize, device)
        elif os.path.isdir(args.input):
            # Process directory of files
            for filename in os.listdir(args.input):
                if filename.endswith(".json"):
                    input_path = os.path.join(args.input, filename)
                    process_single_input(model, input_path, output_dir, args.visualize, device)
        else:
            logger.error(f"Invalid input path: {args.input}")
    else:
        # Process test data from dataloader
        process_test_data(model, config, output_dir, args.batch_size, args.visualize, device)
    
    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main() 