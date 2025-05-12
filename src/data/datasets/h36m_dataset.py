"""
Human3.6M Dataset Implementation
"""
import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Human36MDataset(Dataset):
    """
    PyTorch Dataset for Human3.6M dataset that loads from JSON files containing 2D and 3D keypoints.
    
    Features:
    - Support for both 2D and 3D keypoints
    - Sequence loading for temporal models
    - Configurable joint selection
    - Caching for improved performance
    - Detailed logging and error handling
    - Performance monitoring
    """
    def __init__(
        self, 
        json_files: List[str],
        keypoint_type: str = 'both',  # 'both', '2d', or '3d'
        joint_indices: Optional[List[int]] = None,
        transform=None,
        preload: bool = False,
        sequence_length: int = 1,
        stride: int = 1,
        cache_size: int = 10000,  # Number of samples to cache in memory
        verbose: bool = False,
        log_level: str = "INFO",
        dataset_mean_std=None,  # For normalization
    ):
        """
        Initialize the Human3.6M dataset.
        
        Args:
            json_files: List of paths to JSON files with pose data
            keypoint_type: Type of keypoints to return ('both', '2d', or '3d')
            joint_indices: List of joint indices to include. If None, include all joints
            transform: Optional transform to apply to the data
            preload: If True, load all data into memory
            sequence_length: Number of consecutive frames to return (for sequence models)
            stride: Stride between consecutive sequences
            cache_size: Size of the in-memory LRU cache
            verbose: Enable verbose output
            log_level: Logging level
            dataset_mean_std: Tuple of (mean, std) for normalization
        """
        self.json_files = json_files
        self.keypoint_type = keypoint_type
        self.joint_indices = joint_indices
        self.transform = transform
        self.preload = preload
        self.sequence_length = sequence_length
        self.stride = stride
        self.verbose = verbose
        self.cache_size = cache_size
        self.dataset_mean_std = dataset_mean_std 
        
        # Map frame IDs to (file_idx, frame_idx) for retrieval
        self.frame_mapping = []
        
        # Data cache
        self.data_cache = {}
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.load_times = []
        logger.info(f"Initializing dataset with {len(json_files)} JSON files")
        
        # Validate input files
        self._validate_files()
        # Build frame mapping and preload if requested
        start_time = time.time()
        self._build_frame_mapping()
        logger.info(f"Built frame mapping with {len(self.frame_mapping)} samples in {time.time() - start_time:.2f}s")
        # Apply stride to frame mapping (take every nth sequence)
        self.frame_mapping = self.frame_mapping[::self.stride]
        # Preload data if requested
        if self.preload:
            start_time = time.time()
            self._preload_data()
            logger.info(f"Preloaded data in {time.time() - start_time:.2f}s")
        logger.info(f"Dataset initialized with {len(self.frame_mapping)} sequences")
        
    def _validate_files(self) -> None:
        """Validate that input files exist and are readable."""
        for idx, file_path in enumerate(self.json_files):
            if not os.path.isfile(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Try to open the file to check it's readable
            try:
                with open(file_path, 'r') as _:
                    pass
            except PermissionError:
                logger.error(f"Permission denied accessing: {file_path}")
                raise PermissionError(f"Permission denied accessing: {file_path}")
            except Exception as e:
                logger.error(f"Error accessing file {file_path}: {e}")
                raise
                
    def _build_frame_mapping(self) -> None:
        """
        Build mapping from dataset indices to file and frame indices.
        This allows us to know which file and which frame to load for a given index.
        """
        # Track the number of frames per file for statistics
        frames_per_file = []
        
        for file_idx, json_file in enumerate(self.json_files):
            try:
                logger.debug(f"Scanning {os.path.basename(json_file)}...")
                
                # Memory efficient loading - only load keys first
                frame_ids = self._get_frame_ids(json_file)
                frames_per_file.append(len(frame_ids))
                
                # For each frame that can start a sequence
                valid_frames = 0
                for i in range(0, len(frame_ids) - self.sequence_length + 1):
                    # Check if we can form a valid sequence
                    # For sequences, we need to ensure frames are consecutive
                    if self.sequence_length > 1:
                        # Check if the frames are consecutive
                        # Assuming frame IDs are ordered
                        is_valid_sequence = True
                        for j in range(1, self.sequence_length):
                            if int(frame_ids[i+j]) != int(frame_ids[i]) + j:
                                is_valid_sequence = False
                                break
                                
                        if is_valid_sequence:
                            self.frame_mapping.append((file_idx, frame_ids[i]))
                            valid_frames += 1
                    else:
                        # For single frames, just add the mapping
                        self.frame_mapping.append((file_idx, frame_ids[i]))
                        valid_frames += 1
                    
                logger.debug(f"Added {valid_frames} valid frames/sequences from file {file_idx}")
                
            except Exception as e:
                logger.error(f"Error building frame mapping for {json_file}: {e}")
                # Continue with other files instead of failing completely
                
        # Log statistics about the dataset
        if frames_per_file:
            logger.info(f"Files processed: {len(frames_per_file)}")
            logger.info(f"Total frames across all files: {sum(frames_per_file)}")
            logger.info(f"Average frames per file: {sum(frames_per_file) / len(frames_per_file):.1f}")
            logger.info(f"Valid sequences in dataset: {len(self.frame_mapping)}")
    
    def _get_frame_ids(self, json_file: str) -> List[str]:
        """
        Get all frame IDs from a JSON file efficiently.
        Only loads the keys, not the full data.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        return sorted([k for k in data.keys()])

    def _preload_data(self) -> None:
        """Preload data into memory if requested."""
        logger.info(f"Preloading data from {len(self.json_files)} files...")
        
        for file_idx, json_file in enumerate(self.json_files):
            try:
                start_time = time.time()
                with open(json_file, 'r') as f:
                    self.data_cache[file_idx] = json.load(f)
                
                logger.debug(f"Preloaded file {file_idx} with {len(self.data_cache[file_idx])} frames in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error preloading {json_file}: {e}")
                # Continue with other files
        
        memory_usage_mb = sum(sys.getsizeof(data) for data in self.data_cache.values()) / (1024 * 1024)
        logger.info(f"Preloaded {len(self.data_cache)} files, approximate memory usage: {memory_usage_mb:.2f} MB")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.frame_mapping)
    
    def _load_frame_data(self, file_idx: int, frame_id: str) -> Dict[str, Any]:
        """
        Load data for a specific frame.
        Uses cache if data is preloaded or in LRU cache.
        """
        start_time = time.time()
        
        # Check if data is in cache
        cache_key = f"{file_idx}_{frame_id}"
        if cache_key in self.data_cache:
            self.cache_hits += 1
            frame_data = self.data_cache[cache_key]
            self.load_times.append(time.time() - start_time)
            return frame_data
        
        # If file is preloaded, get data from preloaded file
        if self.preload and file_idx in self.data_cache:
            if frame_id in self.data_cache[file_idx]:
                frame_data = self.data_cache[file_idx][frame_id]
                # Store in frame-level cache for faster access next time
                if len(self.data_cache) < self.cache_size:
                    self.data_cache[cache_key] = frame_data
                self.cache_hits += 1
                self.load_times.append(time.time() - start_time)
                return frame_data
            else:
                # This should not happen if frame_mapping is built correctly
                logger.warning(f"Frame {frame_id} not found in preloaded file {file_idx}")
        
        # Load data from file
        self.cache_misses += 1
        
        try:
            with open(self.json_files[file_idx], 'r') as f:
                file_data = json.load(f)
                
            if frame_id not in file_data:
                logger.error(f"Frame {frame_id} not found in file {self.json_files[file_idx]}")
                raise KeyError(f"Frame {frame_id} not found in file {self.json_files[file_idx]}")
                
            frame_data = file_data[frame_id]
            
            # Store in cache if we have space
            if len(self.data_cache) < self.cache_size:
                self.data_cache[cache_key] = frame_data
                
        except Exception as e:
            logger.error(f"Error loading frame {frame_id} from file {self.json_files[file_idx]}: {e}")
            raise
            
        self.load_times.append(time.time() - start_time)
        return frame_data
    
    def _extract_keypoints(self, frame_data: Dict[str, Any]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Extract 2D and/or 3D keypoints from frame data."""
        # Initialize empty arrays
        keypoints_2d = []
        keypoints_3d = []
        
        try:
            # Process 2D keypoints if needed
            if self.keypoint_type in ['2d', 'both']:
                if 'keypoints_2d' in frame_data:
                    # Get all joint indices
                    if self.joint_indices is None:
                        joint_indices = sorted([int(k) for k in frame_data['keypoints_2d'].keys()])
                    else:
                        joint_indices = self.joint_indices
                    
                    # Extract 2D keypoints
                    for joint_idx in joint_indices:
                        joint_idx_str = str(joint_idx)
                        if joint_idx_str in frame_data['keypoints_2d']:
                            joint_data = frame_data['keypoints_2d'][joint_idx_str]
                            keypoints_2d.append([joint_data['x'], joint_data['y']])
                        else:
                            # Handle missing joints with zeros
                            keypoints_2d.append([0.0, 0.0])
                    
                    keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
                    
                    # Normalize if mean and std are provided
                    if self.dataset_mean_std is not None:
                        mean_2d, std_2d = self.dataset_mean_std[0]
                        keypoints_2d = (keypoints_2d - mean_2d) / std_2d
                else:
                    logger.error("2D keypoints not found in data")
                    raise KeyError("2D keypoints not found in data")
            
            # Process 3D keypoints if needed
            if self.keypoint_type in ['3d', 'both']:
                if 'keypoints_3d' in frame_data:
                    # Get all joint indices
                    if self.joint_indices is None:
                        joint_indices = sorted([int(k) for k in frame_data['keypoints_3d'].keys()])
                    else:
                        joint_indices = self.joint_indices
                    
                    # Extract 3D keypoints
                    for joint_idx in joint_indices:
                        joint_idx_str = str(joint_idx)
                        if joint_idx_str in frame_data['keypoints_3d']:
                            joint_data = frame_data['keypoints_3d'][joint_idx_str]
                            keypoints_3d.append([joint_data['x'], joint_data['y'], joint_data['z']])
                        else:
                            # Handle missing joints with zeros
                            keypoints_3d.append([0.0, 0.0, 0.0])
                    
                    keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
                    
                    # Normalize if mean and std are provided
                    if self.dataset_mean_std is not None:
                        mean_3d, std_3d = self.dataset_mean_std[1]
                        keypoints_3d = (keypoints_3d - mean_3d) / std_3d
                else:
                    logger.error("3D keypoints not found in data")
                    raise KeyError("3D keypoints not found in data")
            
            # Return the appropriate keypoints based on keypoint_type
            if self.keypoint_type == '2d':
                return keypoints_2d
            elif self.keypoint_type == '3d':
                return keypoints_3d
            else:  # 'both'
                return keypoints_2d, keypoints_3d
                
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            raise
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item or sequence from the dataset."""
        # Get the mapping for this index
        try:
            file_idx, start_frame_id = self.frame_mapping[idx]
        except IndexError:
            logger.error(f"Index {idx} out of bounds for dataset with length {len(self.frame_mapping)}")
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.frame_mapping)}")
            
        # Track item fetching time for performance monitoring
        start_time = time.time()
        
        try:
            # Load data for a sequence
            if self.sequence_length > 1:
                # Get data for first frame to determine frame IDs
                data = self._load_frame_data(file_idx, start_frame_id)
                # Initialize sequence arrays
                if self.keypoint_type == 'both':
                    # Extract keypoints for the first frame to determine shape
                    keypoints_2d_first, keypoints_3d_first = self._extract_keypoints(data)
                    seq_2d = np.zeros((self.sequence_length, *keypoints_2d_first.shape), dtype=np.float32)
                    seq_3d = np.zeros((self.sequence_length, *keypoints_3d_first.shape), dtype=np.float32)
                    
                    # Fill first frame
                    seq_2d[0] = keypoints_2d_first
                    seq_3d[0] = keypoints_3d_first
                    
                    # Get consecutive frames if needed
                    for i in range(1, self.sequence_length):
                        next_frame_id = str(int(start_frame_id) + i)
                        
                        try:
                            data = self._load_frame_data(file_idx, next_frame_id)
                            keypoints_2d, keypoints_3d = self._extract_keypoints(data)
                            
                            seq_2d[i] = keypoints_2d
                            seq_3d[i] = keypoints_3d
                        except Exception as e:
                            logger.warning(f"Error loading sequence frame {next_frame_id}: {e}")
                            # Keep zeros for missing frames
                    
                    # Apply transform if provided
                    if self.transform:
                        seq_2d = self.transform(seq_2d)
                        seq_3d = self.transform(seq_3d)
                    
                    # Convert to torch tensors
                    seq_2d = torch.from_numpy(seq_2d)
                    seq_3d = torch.from_numpy(seq_3d)
                    
                    result = {
                        'keypoints_2d': seq_2d, 
                        'keypoints_3d': seq_3d, 
                        'frame_id': start_frame_id,
                        'file_idx': file_idx
                    }
                else:  # '2d' or '3d'
                    # Extract keypoints for the first frame to determine shape
                    keypoints_first = self._extract_keypoints(data)
                    seq = np.zeros((self.sequence_length, *keypoints_first.shape), dtype=np.float32)
                    seq[0] = keypoints_first
                    
                    # Get consecutive frames if needed
                    for i in range(1, self.sequence_length):
                        next_frame_id = str(int(start_frame_id) + i)
                        
                        try:
                            data = self._load_frame_data(file_idx, next_frame_id)
                            keypoints = self._extract_keypoints(data)
                            seq[i] = keypoints
                        except Exception as e:
                            logger.warning(f"Error loading sequence frame {next_frame_id}: {e}")
                            # Keep zeros for missing frames
                    
                    # Apply transform if provided
                    if self.transform:
                        seq = self.transform(seq)
                    
                    # Convert to torch tensor
                    seq = torch.from_numpy(seq)
                    
                    result = {
                        f'keypoints_{self.keypoint_type}': seq, 
                        'frame_id': start_frame_id,
                        'file_idx': file_idx
                    }
                    
            else:  # Single frame
                # Load data
                data = self._load_frame_data(file_idx, start_frame_id)
                # Extract keypoints
                keypoints = self._extract_keypoints(data)
                # Apply transform if provided

                if self.transform:
                    if isinstance(keypoints, tuple):
                        keypoints = tuple(self.transform(k) for k in keypoints)
                    else:
                        keypoints = self.transform(keypoints)
                # Convert to torch tensor
                if isinstance(keypoints, tuple):
                    keypoints_2d, keypoints_3d = keypoints
                    keypoints_2d = torch.from_numpy(keypoints_2d)
                    keypoints_3d = torch.from_numpy(keypoints_3d)
                    
                    result = {
                        'keypoints_2d': keypoints_2d.to(torch.float32), 
                        'keypoints_3d': keypoints_3d.to(torch.float32), 
                        'frame_id': start_frame_id,
                        'file_idx': file_idx
                    }
                else:
                    keypoints = torch.from_numpy(keypoints)
                    result = {
                        f'keypoints_{self.keypoint_type}': keypoints.to(torch.float32), 
                        'frame_id': start_frame_id,
                        'file_idx': file_idx
                    }
            
            # Add metadata if needed
            result['idx'] = idx
            # Log item fetch time if requested
            if self.verbose and idx % 1000 == 0:
                fetch_time = time.time() - start_time
                
                # Log cache stats occasionally
                if idx % 5000 == 0:
                    total = self.cache_hits + self.cache_misses
                    if total > 0:
                        hit_rate = self.cache_hits / total * 100
                        if self.load_times:
                            avg_load_time = sum(self.load_times) / len(self.load_times)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            # Return a dummy item if there's an error to avoid crashing the training
            # This can be useful in production when a few corrupt samples shouldn't crash the entire run
            # For debugging, you might want to raise the exception instead
            if self.keypoint_type == 'both':
                # Create zero tensors with appropriate shapes
                dummy_2d = torch.zeros((self.sequence_length if self.sequence_length > 1 else 1, 133, 2), dtype=torch.float32)
                dummy_3d = torch.zeros((self.sequence_length if self.sequence_length > 1 else 1, 133, 3), dtype=torch.float32)
                return {
                    'keypoints_2d': dummy_2d,
                    'keypoints_3d': dummy_3d,
                    'frame_id': '-1',  # Dummy frame ID
                    'file_idx': -1,
                    'is_dummy': True,  # Flag to indicate this is a dummy item
                    'error': str(e)
                }
            else:
                dims = 2 if self.keypoint_type == '2d' else 3
                dummy = torch.zeros((self.sequence_length if self.sequence_length > 1 else 1, 133, dims), dtype=torch.float32)
                return {
                    f'keypoints_{self.keypoint_type}': dummy,
                    'frame_id': '-1',  # Dummy frame ID
                    'file_idx': -1,
                    'is_dummy': True,  # Flag to indicate this is a dummy item
                    'error': str(e)
                }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return statistics about cache usage."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
        avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_accesses': total,
            'hit_rate': hit_rate,
            'avg_load_time_ms': avg_load_time * 1000,
            'cache_size': len(self.data_cache),
            'max_cache_size': self.cache_size
        }
    
    def compute_dataset_stats(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Compute mean and standard deviation of the dataset.
        Returns ((mean_2d, std_2d), (mean_3d, std_3d))
        """
        logger.info("Computing dataset statistics...")
        
        # Accumulate statistics
        sum_2d = np.zeros((133, 2))
        sum_3d = np.zeros((133, 3))
        sum_sq_2d = np.zeros((133, 2))
        sum_sq_3d = np.zeros((133, 3))
        count = 0
        
        # Sample a subset of the dataset for efficiency
        indices = np.random.choice(len(self), min(10000, len(self)), replace=False)
        
        for idx in indices:
            sample = self[idx]
            
            if 'keypoints_2d' in sample:
                kp_2d = sample['keypoints_2d'].numpy() if isinstance(sample['keypoints_2d'], torch.Tensor) else sample['keypoints_2d']
                if len(kp_2d.shape) == 3:  # If sequence data
                    kp_2d = kp_2d[0]  # Just use the first frame
                sum_2d += kp_2d
                sum_sq_2d += kp_2d**2
                
            if 'keypoints_3d' in sample:
                kp_3d = sample['keypoints_3d'].numpy() if isinstance(sample['keypoints_3d'], torch.Tensor) else sample['keypoints_3d']
                if len(kp_3d.shape) == 3:  # If sequence data
                    kp_3d = kp_3d[0]  # Just use the first frame
                sum_3d += kp_3d
                sum_sq_3d += kp_3d**2
                
            count += 1
            
        # Compute mean and std
        mean_2d = sum_2d / count
        mean_3d = sum_3d / count
        
        std_2d = np.sqrt(sum_sq_2d / count - mean_2d**2)
        std_3d = np.sqrt(sum_sq_3d / count - mean_3d**2)
        
        logger.info(f"Computed statistics over {count} samples")
        
        # Set small values to 1 to avoid division by zero
        std_2d[std_2d < 1e-6] = 1.0
        std_3d[std_3d < 1e-6] = 1.0
        
        return ((mean_2d, std_2d), (mean_3d, std_3d))

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (f"Human36MDataset(files: {len(self.json_files)}, "
                f"samples: {len(self.frame_mapping)}, "
                f"keypoint_type: {self.keypoint_type}, "
                f"sequence_length: {self.sequence_length}, "
                f"preload: {self.preload})")


# Import dependencies only when needed to avoid import errors
import sys
# This allows the file to be imported even if these imports fail
try:
    from torch.utils.data import DataLoader
except ImportError:
    logger.warning("Could not import DataLoader from torch.utils.data")

def get_human36m_dataloader(
    json_files: List[str],
    batch_size: int = 32,
    keypoint_type: str = 'both',
    joint_indices: Optional[List[int]] = None,
    transform=None,
    preload: bool = False,
    sequence_length: int = 1,
    stride: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the Human3.6M dataset.
    
    Args:
        json_files: List of paths to JSON files
        batch_size: Batch size for the DataLoader
        keypoint_type: Type of keypoints to return ('both', '2d', or '3d')
        joint_indices: List of joint indices to include. If None, include all joints
        transform: Optional transform to apply to the data
        preload: If True, load all data into memory
        sequence_length: Number of consecutive frames to return
        stride: Stride between consecutive sequences
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        persistent_workers: Keep worker processes alive between batches
    
    Returns:
        DataLoader: PyTorch DataLoader for the Human3.6M dataset
    """
    dataset = Human36MDataset(
        json_files=json_files,
        keypoint_type=keypoint_type,
        joint_indices=joint_indices,
        transform=transform,
        preload=preload,
        sequence_length=sequence_length,
        stride=stride
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    return dataloader 