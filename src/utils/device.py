"""
Device utility functions for model training and inference.

This module provides functions to select the appropriate device 
(CUDA GPU, MPS on Mac, or CPU) for model training and inference.
"""
import torch
import platform
import logging

logger = logging.getLogger(__name__)

def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    This function tries to use:
    1. CUDA if available (with specified GPU ID)
    2. MPS (Metal Performance Shaders) if on Mac with M-series chip
    3. CPU as fallback
    
    Args:
        gpu_id: Index of GPU to use if CUDA is available
        
    Returns:
        torch.device: The selected device
    """
    # Check for CUDA first
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(gpu_id)}")
        return device
    
    # Check for MPS (Apple Silicon GPU) if on Mac
    # MPS is available in PyTorch 1.12+ on macOS 12.3+
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon GPU)")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device

def print_device_info() -> None:
    """
    Print detailed information about available devices.
    """
    print("\n=== Device Information ===")
    
    # CUDA information
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        print(f"CUDA is available with {cuda_count} device(s):")
        for i in range(cuda_count):
            print(f"  - {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"    - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available")
    
    # MPS information
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available")
            print("  - macOS version:", platform.mac_ver()[0])
            print("  - Device: Apple Silicon (M-series)")
        else:
            print("MPS is not available")
            if torch.backends.mps.is_built():
                print("  - MPS is built in PyTorch but not available on this system")
            else:
                print("  - MPS is not built in this PyTorch installation")
    
    print("\nSelected device:", get_device())
    print("=========================\n") 