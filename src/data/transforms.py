"""
Data transforms for human pose data.

This module contains transforms for preprocessing and augmenting human pose data.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize with list of transforms.
        
        Args:
            transforms: List of transform functions/objects to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, *args):
        """
        Apply all transforms in sequence.
        
        Args:
            *args: Arguments to pass to each transform
        
        Returns:
            Transformed data
        """
        for transform in self.transforms:
            if len(args) == 1:
                args = (transform(args[0]),)
            else:
                args = transform(*args)
        
        return args[0] if len(args) == 1 else args


class Normalize:
    """Normalize keypoints by mean and standard deviation."""
    
    def __init__(
        self, 
        mean: Optional[np.ndarray] = None, 
        std: Optional[np.ndarray] = None,
    ):
        """
        Initialize with normalization parameters.
        
        Args:
            mean: Mean of 2D or 3D keypoints, shape [num_joints, 2/3]
            std: Standard deviation of 2D or 3D keypoints, shape [num_joints, 2/3]
        """
        self.mean = mean
        self.std = std
        
        # Ensure std doesn't have zeros to avoid division by zero
        if self.std is not None:
            self.std = np.where(self.std < 1e-6, 1.0, self.std)
    
    def __call__(self, keypoints=None):
        """
        Normalize keypoints.
        
        Args:
            keypoints: 2D or 3D keypoints, shape [..., num_joints, 2/3]
            
        Returns:
            Normalized keypoints
        """
        if keypoints is not None and self.mean is not None and self.std is not None:
            keypoints = (keypoints - self.mean) / self.std
            
        return keypoints


class RandomFlip:
    """Randomly flip poses horizontally."""
    
    def __init__(self, probability: float = 0.5, flip_indices: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize with flip parameters.
        
        Args:
            probability: Probability of flipping
            flip_indices: List of joint index pairs to swap when flipping
                          Each pair is (left_idx, right_idx)
        """
        self.probability = probability
        self.flip_indices = flip_indices
        
    def __call__(self, keypoints=None):
        """
        Randomly flip keypoints.
        
        Args:
            keypoints: 2D or 3D keypoints, shape [..., num_joints, 2/3]
            
        Returns:
            Flipped keypoints
        """
        # Determine if we should flip
        if np.random.random() >= self.probability:
            return keypoints
            
        # Flip 2D keypoints
        if keypoints is not None:
            # Get a copy to avoid modifying the original
            keypoints = keypoints.copy()
            
            # Flip x coordinates
            keypoints[..., 0] = -keypoints[..., 0]
            
            # Swap left and right joints if flip indices are provided
            if self.flip_indices is not None:
                for left_idx, right_idx in self.flip_indices:
                    keypoints[..., left_idx, :], keypoints[..., right_idx, :] = \
                        keypoints[..., right_idx, :].copy(), keypoints[..., left_idx, :].copy()
        return keypoints


class RandomRotation:
    """Randomly rotate poses in the horizontal plane."""
    
    def __init__(self, max_rotation_degrees: float = 15.0):
        """
        Initialize with rotation parameters.
        
        Args:
            max_rotation_degrees: Maximum rotation in degrees
        """
        self.max_rotation_degrees = max_rotation_degrees
        
    def __call__(self, keypoints=None):
        """
        Randomly rotate keypoints.
        
        Args:
            keypoints_2d: 2D keypoints, shape [..., num_joints, 2]
            keypoints_3d: 3D keypoints, shape [..., num_joints, 3]
            
        Returns:
            Rotated keypoints
        """
        # Sample a random rotation angle
        angle_degrees = np.random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        angle_radians = np.deg2rad(angle_degrees)
        # Create rotation matrix for 2D
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        rotation_matrix_2d = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Create rotation matrix for 3D (rotate around y-axis)
        rotation_matrix_3d = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        # Apply rotation to 2D keypoints
        if keypoints is not None:
            original_shape = keypoints.shape
            dims = 2 if keypoints.shape[-1] == 2 else 3
            rotation_matrix = rotation_matrix_2d if dims == 2 else rotation_matrix_3d
            keypoints_reshaped = keypoints.reshape(-1, dims)
            # Apply rotation
            keypoints_rotated = np.matmul(keypoints_reshaped, rotation_matrix.T)
            # Restore original shape
            keypoints = keypoints_rotated.reshape(original_shape)
        return keypoints


class RandomScale:
    """Randomly scale poses."""
    
    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2):
        """
        Initialize with scale parameters.
        
        Args:
            min_scale: Minimum scale factor
            max_scale: Maximum scale factor
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, keypoints=None):
        """
        Randomly scale keypoints.
        
        Args:
            keypoints: 2D or 3D keypoints, shape [..., num_joints, 2/3]
            
        Returns:
            Scaled keypoints
        """
       
        # Sample a random scale factor
        scale = np.random.uniform(self.min_scale, self.max_scale)
        
        # Apply scaling to keypoints
        if keypoints is not None:
            keypoints = keypoints * scale
            
        return keypoints


class RandomJitter:
    """Add random noise to poses."""
    
    def __init__(
        self, 
        max_jitter_2d: float = 0.02, 
        max_jitter_3d: float = 0.02,
        noise_type: str = "gaussian"  # gaussian or uniform
    ):
        """
        Initialize with jitter parameters.
        
        Args:
            max_jitter_2d: Maximum jitter magnitude for 2D keypoints (as fraction of pose size)
            max_jitter_3d: Maximum jitter magnitude for 3D keypoints (as fraction of pose size)
            noise_type: Type of noise to add ('gaussian' or 'uniform')
        """
        self.max_jitter_2d = max_jitter_2d
        self.max_jitter_3d = max_jitter_3d
        self.noise_type = noise_type.lower()
        
    def __call__(self, keypoints_2d=None, keypoints_3d=None):
        """
        Add random noise to keypoints.
        
        Args:
            keypoints_2d: 2D keypoints, shape [..., num_joints, 2]
            keypoints_3d: 3D keypoints, shape [..., num_joints, 3]
            
        Returns:
            Jittered keypoints
        """
        # Add jitter to 2D keypoints
        if keypoints_2d is not None:
            # Calculate pose size
            if len(keypoints_2d.shape) == 3:  # Sequence data
                pose_size_2d = np.max(keypoints_2d[0]) - np.min(keypoints_2d[0])
            else:
                pose_size_2d = np.max(keypoints_2d) - np.min(keypoints_2d)
                
            jitter_std_2d = pose_size_2d * self.max_jitter_2d
            
            # Generate noise
            if self.noise_type == "gaussian":
                noise_2d = np.random.normal(0, jitter_std_2d, keypoints_2d.shape)
            else:  # uniform
                noise_2d = np.random.uniform(-jitter_std_2d, jitter_std_2d, keypoints_2d.shape)
                
            keypoints_2d = keypoints_2d + noise_2d
        
        # Add jitter to 3D keypoints
        if keypoints_3d is not None:
            # Calculate pose size
            if len(keypoints_3d.shape) == 3:  # Sequence data
                pose_size_3d = np.max(keypoints_3d[0]) - np.min(keypoints_3d[0])
            else:
                pose_size_3d = np.max(keypoints_3d) - np.min(keypoints_3d)
                
            jitter_std_3d = pose_size_3d * self.max_jitter_3d
            
            # Generate noise
            if self.noise_type == "gaussian":
                noise_3d = np.random.normal(0, jitter_std_3d, keypoints_3d.shape)
            else:  # uniform
                noise_3d = np.random.uniform(-jitter_std_3d, jitter_std_3d, keypoints_3d.shape)
                
            keypoints_3d = keypoints_3d + noise_3d
        
        if keypoints_2d is None:
            return keypoints_3d
        if keypoints_3d is None:
            return keypoints_2d
            
        return keypoints_2d, keypoints_3d


class CenterAtRoot:
    """Center the pose at the root joint."""
    
    def __init__(self, root_joint_idx: int = 0):
        """
        Initialize with root joint index.
        
        Args:
            root_joint_idx: Index of the root joint
        """
        self.root_joint_idx = root_joint_idx
        
    def __call__(self, keypoints=None):
        """
        Center keypoints at the root joint.
        
        Args:
            keypoints: 2D or 3D keypoints, shape [..., num_joints, 2/3]
            
        Returns:
            Centered keypoints
        """
        # Center keypoints
        if keypoints is not None:
            if len(keypoints.shape) == 3:  # Sequence data
                root = keypoints[:, self.root_joint_idx:self.root_joint_idx+1, :]

            else:
                root = keypoints[self.root_joint_idx:self.root_joint_idx+1, :]
            keypoints = keypoints - root

        return keypoints


# Create default transform sets for different use cases
def get_train_transforms(
    mean=None, std=None,
    flip_indices=None, 
    flip_probability=0.5,
    max_rotation=15.0, 
    scale_range=(0.8, 1.2),
    use_jitter=True,
    root_joint_idx=0
):
    """
    Get standard transforms for training.
    
    Args:
        mean: Mean of 2D or 3D keypoints
        std: Standard deviation of 2D or 3D keypoints
        flip_indices: List of joint index pairs to swap when flipping
        max_rotation: Maximum rotation in degrees
        scale_range: Range of scaling factors (min, max)
        use_jitter: Whether to add random jitter
        root_joint_idx: Index of the root joint
        
    Returns:
        Compose: Composition of transforms
    """
    transforms = []
    
    # Center at root joint
    transforms.append(CenterAtRoot(root_joint_idx=root_joint_idx))
    
    # Normalize if stats are provided
    if mean is not None and std is not None:
        transforms.append(Normalize(mean=mean, std=std))
    
    # Data augmentation
    transforms.append(RandomFlip(probability=flip_probability, flip_indices=flip_indices))
    transforms.append(RandomRotation(max_rotation_degrees=max_rotation))
    transforms.append(RandomScale(min_scale=scale_range[0], max_scale=scale_range[1]))
    
    if use_jitter:
        transforms.append(RandomJitter(max_jitter_2d=0.02, max_jitter_3d=0.02))
    
    
    return Compose(transforms)


def get_val_transforms(
    mean=None, std=None, 
    root_joint_idx=0
):
    """
    Get standard transforms for validation.
    
    Args:
        mean: Mean of 2D keypoints
        std: Standard deviation of 2D keypoints
        root_joint_idx: Index of the root joint
        
    Returns:
        Compose: Composition of transforms
    """
    transforms = []
    
    # Center at root joint
    transforms.append(CenterAtRoot(root_joint_idx=root_joint_idx))
    
    # Normalize if stats are provided
    if mean is not None and std is not None:
        transforms.append(Normalize(mean=mean, std=std))
    
    
    return Compose(transforms) 