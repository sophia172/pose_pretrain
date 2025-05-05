"""
Data loader factories and utilities for Human Pose Estimation.

This module provides factory functions for creating PyTorch DataLoaders
for different datasets and configurations.
"""
import os
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import DataLoader, ConcatDataset

from .datasets import Human36MDataset
from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def load_dataset_config(config_file: str) -> Dict[str, Any]:
    """
    Load dataset configuration from YAML file.
    
    Args:
        config_file: Path to YAML config file
        
    Returns:
        Dict: Dataset configuration
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load dataset config from {config_file}: {e}")
        raise


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve a path which might contain variables like ${paths.base_dir}.
    
    Args:
        path: Path string possibly containing variables
        base_dir: Base directory to use for relative paths
        
    Returns:
        str: Resolved path
    """
    # If path is absolute, return it
    if os.path.isabs(path):
        return path
        
    # If path contains variables, resolve them
    if '${' in path:
        # Simple variable replacement
        if '${paths.base_dir}' in path:
            path = path.replace('${paths.base_dir}', base_dir or '')
            
    # If path is relative and base_dir is provided, join them
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
        
    return path


def get_human36m_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders for Human3.6M dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple: (train_loader, val_loader)
    """
    logger.info("Creating Human3.6M DataLoaders")
    
    # Get dataset config
    try:
        dataset_config = load_dataset_config(config['data']['config_file'])
    except Exception as e:
        logger.error(f"Failed to load dataset config: {e}")
        dataset_config = {}
    
    # Resolve paths
    dataset_dir = config['paths'].get('dataset', '')
    # Get train file paths
    train_json_files = []
    train_files = dataset_config['paths'].get('train_json_files', '')

    for file in train_files:
        file = os.path.join(dataset_dir, 
                            dataset_config['paths'].get('base_dir', ''),
                            dataset_config['paths'].get('annotations_dir', ''),
                            file)
        file = file.strip()
        if os.path.exists(file):
            train_json_files.append(file)
    
    # Get validation file path
    val_json_files = []
    val_files = dataset_config['paths'].get('val_json_files', '')
    for file in val_files:
        file = os.path.join(dataset_dir,
                            dataset_config['paths'].get('base_dir', ''),
                            dataset_config['paths'].get('annotations_dir', ''), 
                            file)
        file = file.strip()
        if os.path.exists(file):
            val_json_files.append(file)
    
    # Check that we have valid files
    if not train_json_files:
        logger.error("No valid train JSON files found")
        raise FileNotFoundError("No valid train JSON files found")
        
    logger.info(f"Found {len(train_json_files)} train JSON files")
    if val_json_files:
        logger.info(f"Using validation files: {val_json_files}")
    else:
        logger.warning("No validation file found")
    
    # Get dataset parameters
    batch_size = config['data'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['data'].get('pin_memory', True)
    shuffle = config['data'].get('shuffle', True)
    sequence_length = config['data'].get('sequence_length', 1)
    stride = config['data'].get('stride', 1)
    keypoint_type = config['data'].get('keypoint_type', 'both')
    preload = config['data'].get('preload', False)
    
    # Set up transforms
    flip_indices = dataset_config.get('augmentation', {}).get('flip_indices', None)
    
    # Create train dataset
    train_transform = get_train_transforms(
        flip_indices=flip_indices,
        max_rotation=dataset_config.get('augmentation', {}).get('rotate', {}).get('max_rotation', 15.0),
        scale_range=(
            dataset_config.get('augmentation', {}).get('scale', {}).get('min_scale', 0.8),
            dataset_config.get('augmentation', {}).get('scale', {}).get('max_scale', 1.2)
        ),
        use_jitter=dataset_config.get('augmentation', {}).get('jitter', {}).get('enabled', True),
        root_joint_idx=0  # Assuming 0 is the root joint
    )
    
    # Create validation transform
    val_transform = get_val_transforms(root_joint_idx=0)
    
    # Create train dataset
    logger.info("Creating train dataset")
    train_dataset = Human36MDataset(
        json_files=train_json_files,
        keypoint_type=keypoint_type,
        transform=train_transform,
        preload=preload,
        sequence_length=sequence_length,
        stride=stride,
        verbose=config.get('experiment', {}).get('debug', False)
    )
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    # Create validation dataset and loader if validation file is available
    val_loader = None
    if val_json_files:
        logger.info("Creating validation dataset")
        val_dataset = Human36MDataset(
            json_files=val_json_files,
            keypoint_type=keypoint_type,
            transform=val_transform,
            preload=preload,
            sequence_length=sequence_length,
            stride=stride,
            verbose=config.get('experiment', {}).get('debug', False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_loader:
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    return train_loader, val_loader


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Factory function for creating dataloaders based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple: (train_loader, val_loader)
    """
    dataset_name = config['data'].get('dataset', 'h36m')
    
    if dataset_name.lower() == 'h36m':
        return get_human36m_dataloaders(config)
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_stats(config: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute and return dataset statistics.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple: ((mean_2d, std_2d), (mean_3d, std_3d))
    """
    logger.info("Computing dataset statistics")
    
    # Create a temporary dataset for computing statistics
    dataset_name = config['data'].get('dataset', 'h36m')
    
    if dataset_name.lower() == 'h36m':
        try:
            # Get dataset config
            dataset_config = load_dataset_config(config['data']['config_file'])
        except Exception as e:
            logger.error(f"Failed to load dataset config: {e}")
            dataset_config = {}
        
        # Resolve paths
        base_dir = config['paths'].get('dataset', '')
        
        # Get train file paths
        train_json_files = []
        train_files_str = config['data'].get('train_split_file', '')
        if train_files_str:
            train_files = train_files_str.split(',')
            for file in train_files:
                file = file.strip()
                if file:
                    for root_dir in [dataset_config.get('paths', {}).get('annotations_dir', ''), base_dir]:
                        path = resolve_path(os.path.join(root_dir, file), base_dir)
                        if os.path.exists(path):
                            train_json_files.append(path)
                            break
        
        if not train_json_files and 'train_json_files' in dataset_config.get('paths', {}):
            # Try to get from dataset config
            for file in dataset_config['paths']['train_json_files']:
                path = resolve_path(file, base_dir)
                if os.path.exists(path):
                    train_json_files.append(path)
        
        # Use only the first file for efficiency
        if train_json_files:
            stats_dataset = Human36MDataset(
                json_files=[train_json_files[0]],
                keypoint_type='both',
                preload=False,
                sequence_length=1,
                verbose=False
            )
            
            # Compute statistics
            (mean_2d, std_2d), (mean_3d, std_3d) = stats_dataset.compute_dataset_stats()
            
            # Convert to torch tensors
            mean_2d = torch.from_numpy(mean_2d).float()
            std_2d = torch.from_numpy(std_2d).float()
            mean_3d = torch.from_numpy(mean_3d).float()
            std_3d = torch.from_numpy(std_3d).float()
            
            return (mean_2d, std_2d), (mean_3d, std_3d)
        else:
            logger.error("No valid train files found for computing statistics")
            raise FileNotFoundError("No valid train files found for computing statistics")
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'paths': {
            'dataset': 'ML dataset'
        },
        'data': {
            'dataset': 'h36m',
            'config_file': 'config/data/h36m.yaml',
            'train_split_file': 'train_part1.json',
            'val_split_file': 'test_2d.json',
            'batch_size': 32,
            'num_workers': 4,
            'shuffle': True,
            'sequence_length': 1,
            'keypoint_type': 'both'
        },
        'experiment': {
            'debug': True
        }
    }
    
    try:
        train_loader, val_loader = get_dataloaders(config)
        print(f"Created train loader with {len(train_loader)} batches")
        if val_loader:
            print(f"Created val loader with {len(val_loader)} batches")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        
    try:
        (mean_2d, std_2d), (mean_3d, std_3d) = get_dataset_stats(config)
        print(f"Computed statistics for dataset")
        print(f"2D mean shape: {mean_2d.shape}, std shape: {std_2d.shape}")
        print(f"3D mean shape: {mean_3d.shape}, std shape: {std_3d.shape}")
    except Exception as e:
        print(f"Error computing dataset statistics: {e}") 