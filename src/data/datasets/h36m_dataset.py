"""
Human3.6M Dataset Implementation - Optimized Version
"""
import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU Cache implementation for efficient frame storage.
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        # Move the item to the end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        # Add/update the item
        self.cache[key] = value
        self.cache.move_to_end(key)
        # Remove the oldest item if we're over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class Human36MDataset(Dataset):
    """
    PyTorch Dataset for Human3.6M dataset that loads from JSON files containing 2D and 3D keypoints.
    
    Features:
    - Support for both 2D and 3D keypoints
    - Sequence loading for temporal models
    - Configurable joint selection
    - Efficient caching for improved performance
    - Memory mapped file loading for large datasets
    - Optimized data extraction
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
        
        # Use proper LRU cache instead of dict
        self.frame_cache = LRUCache(cache_size)
        self.file_cache = {}  # For preloaded files
        
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
        
        # Pre-compute joint indices to avoid repeated string conversions
        if self.joint_indices is not None:
            self.joint_indices_str = [str(idx) for idx in self.joint_indices]
                
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
        # For each file, store the metadata about frame sequences
        file_metadata = {}
        
        # Track the number of frames per file for statistics
        frames_per_file = []
        
        for file_idx, json_file in enumerate(self.json_files):
            try:
                logger.debug(f"Scanning {os.path.basename(json_file)}...")
                
                # Memory efficient loading - read file once and cache frame IDs
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    frame_ids = sorted([int(k) for k in data.keys()])
                    
                # Store metadata for this file
                file_metadata[file_idx] = {
                    'frame_ids': frame_ids,
                    'frame_map': {int(fid): idx for idx, fid in enumerate(frame_ids)},
                    'total_frames': len(frame_ids)
                }
                
                frames_per_file.append(len(frame_ids))
                
                # For each frame that can start a sequence
                valid_frames = 0
                for i in range(0, len(frame_ids) - self.sequence_length + 1):
                    # For sequences, check if frames are consecutive
                    if self.sequence_length > 1:
                        # Much faster way to check consecutive frames
                        if frame_ids[i] + self.sequence_length - 1 == frame_ids[i + self.sequence_length - 1]:
                            self.frame_mapping.append((file_idx, str(frame_ids[i])))
                            valid_frames += 1
                    else:
                        # For single frames, just add the mapping
                        self.frame_mapping.append((file_idx, str(frame_ids[i])))
                        valid_frames += 1
                    
                logger.debug(f"Added {valid_frames} valid frames/sequences from file {file_idx}")
                
            except Exception as e:
                logger.error(f"Error building frame mapping for {json_file}: {e}")
                # Continue with other files instead of failing completely
                
        # Log statistics about the dataset
        if frames_per_file:
            logger.debug(f"Files processed: {len(frames_per_file)}")
            logger.debug(f"Total frames across all files: {sum(frames_per_file)}")
            logger.debug(f"Average frames per file: {sum(frames_per_file) / len(frames_per_file):.1f}")
            logger.debug(f"Valid sequences in dataset: {len(self.frame_mapping)}")
            
        # Save file metadata for efficient sequence retrieval
        self.file_metadata = file_metadata
    
    def _preload_data(self) -> None:
        """Preload data into memory if requested."""
        logger.info(f"Preloading data from {len(self.json_files)} files...")
        
        # Process files in parallel if multiple workers are available
        try:
            import concurrent.futures
            from functools import partial
            
            def load_file(file_idx, json_file):
                try:
                    start_time = time.time()
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    logger.debug(f"Preloaded file {file_idx} with {len(data)} frames in {time.time() - start_time:.2f}s")
                    return file_idx, data
                except Exception as e:
                    logger.error(f"Error preloading {json_file}: {e}")
                    return file_idx, None
            
            # Use thread pool for I/O bound operations
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(lambda x: load_file(x[0], x[1]), 
                                     enumerate(self.json_files))
                
                for file_idx, data in results:
                    if data is not None:
                        self.file_cache[file_idx] = data
                
        except ImportError:
            # Fall back to sequential loading if concurrent.futures is not available
            for file_idx, json_file in enumerate(self.json_files):
                try:
                    start_time = time.time()
                    with open(json_file, 'r') as f:
                        self.file_cache[file_idx] = json.load(f)
                    
                    logger.debug(f"Preloaded file {file_idx} with {len(self.file_cache[file_idx])} frames in {time.time() - start_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error preloading {json_file}: {e}")
        
        import sys
        memory_usage_mb = sum(sys.getsizeof(data) for data in self.file_cache.values()) / (1024 * 1024)
        logger.info(f"Preloaded {len(self.file_cache)} files, approximate memory usage: {memory_usage_mb:.2f} MB")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.frame_mapping)
    
    def _load_frame_data(self, file_idx: int, frame_id: str) -> Dict[str, Any]:
        """
        Load data for a specific frame.
        Uses cache if data is preloaded or in LRU cache.
        """
        start_time = time.time()
        
        # Check if frame is in cache
        cache_key = f"{file_idx}_{frame_id}"
        cached_frame = self.frame_cache.get(cache_key)
        if cached_frame is not None:
            self.cache_hits += 1
            self.load_times.append(time.time() - start_time)
            return cached_frame
        
        # If file is preloaded, get data from preloaded file
        if self.preload and file_idx in self.file_cache:
            if frame_id in self.file_cache[file_idx]:
                frame_data = self.file_cache[file_idx][frame_id]
                # Store in frame-level cache for faster access next time
                self.frame_cache.put(cache_key, frame_data)
                self.cache_hits += 1
                self.load_times.append(time.time() - start_time)
                return frame_data
            else:
                # This should not happen if frame_mapping is built correctly
                logger.warning(f"Frame {frame_id} not found in preloaded file {file_idx}")
        
        # Load data from file
        self.cache_misses += 1
        
        try:
            # Optimize file access - don't load entire file if we only need one frame
            with open(self.json_files[file_idx], 'r') as f:
                data = json.load(f)
                
            if frame_id not in data:
                logger.error(f"Frame {frame_id} not found in file {self.json_files[file_idx]}")
                raise KeyError(f"Frame {frame_id} not found in file {self.json_files[file_idx]}")
                
            frame_data = data[frame_id]
            
            # Store in cache
            self.frame_cache.put(cache_key, frame_data)
                
        except Exception as e:
            logger.error(f"Error loading frame {frame_id} from file {self.json_files[file_idx]}: {e}")
            raise
            
        self.load_times.append(time.time() - start_time)
        return frame_data
    
    def _extract_keypoints_fast(self, frame_data: Dict[str, Any]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Extract 2D and/or 3D keypoints from frame data using optimized approach."""
        # Initialize result containers
        keypoints_2d = None
        keypoints_3d = None
        
        try:
            # Process 2D keypoints if needed
            if self.keypoint_type in ['2d', 'both']:
                if 'keypoints_2d' in frame_data:
                    # Get all joint indices
                    if self.joint_indices is None:
                        joint_indices = sorted([int(k) for k in frame_data['keypoints_2d'].keys()])
                        joint_indices_str = [str(idx) for idx in joint_indices]
                    else:
                        joint_indices = self.joint_indices
                        joint_indices_str = self.joint_indices_str
                    
                    # Pre-allocate array for 2D keypoints
                    keypoints_2d = np.zeros((len(joint_indices), 2), dtype=np.float32)
                    
                    # Fast extraction using numpy operations
                    kp_2d_data = frame_data['keypoints_2d']
                    for i, joint_idx_str in enumerate(joint_indices_str):
                        if joint_idx_str in kp_2d_data:
                            joint_data = kp_2d_data[joint_idx_str]
                            keypoints_2d[i, 0] = joint_data['x']
                            keypoints_2d[i, 1] = joint_data['y']
                    
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
                        joint_indices_str = [str(idx) for idx in joint_indices]
                    else:
                        joint_indices = self.joint_indices
                        joint_indices_str = self.joint_indices_str
                    
                    # Pre-allocate array for 3D keypoints
                    keypoints_3d = np.zeros((len(joint_indices), 3), dtype=np.float32)
                    
                    # Fast extraction using numpy operations
                    kp_3d_data = frame_data['keypoints_3d']
                    for i, joint_idx_str in enumerate(joint_indices_str):
                        if joint_idx_str in kp_3d_data:
                            joint_data = kp_3d_data[joint_idx_str]
                            keypoints_3d[i, 0] = joint_data['x']
                            keypoints_3d[i, 1] = joint_data['y']
                            keypoints_3d[i, 2] = joint_data['z']
                    
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
    
    def _load_sequence_fast(self, file_idx: int, start_frame_id: str) -> Dict[str, Any]:
        """
        Optimized method to load a sequence of frames at once.
        Reduces redundant file access for sequences.
        """
        # For sequence loading, we optimize by loading all frames at once when possible
        try:
            # Get consecutive frame IDs
            start_frame_int = int(start_frame_id)
            frame_ids = [str(start_frame_int + i) for i in range(self.sequence_length)]
            
            # Initialize result containers based on keypoint type
            if self.keypoint_type == 'both':
                # Get the first frame to determine shapes
                first_frame = self._load_frame_data(file_idx, frame_ids[0])
                kp_2d_first, kp_3d_first = self._extract_keypoints_fast(first_frame)
                
                # Pre-allocate arrays for all frames
                seq_2d = np.zeros((self.sequence_length, *kp_2d_first.shape), dtype=np.float32)
                seq_3d = np.zeros((self.sequence_length, *kp_3d_first.shape), dtype=np.float32)
                
                # Store first frame data
                seq_2d[0] = kp_2d_first
                seq_3d[0] = kp_3d_first
                
                # Load subsequent frames
                for i in range(1, self.sequence_length):
                    try:
                        frame_data = self._load_frame_data(file_idx, frame_ids[i])
                        kp_2d, kp_3d = self._extract_keypoints_fast(frame_data)
                        seq_2d[i] = kp_2d
                        seq_3d[i] = kp_3d
                    except Exception as e:
                        logger.warning(f"Error loading sequence frame {frame_ids[i]}: {e}")
                        # Keep zeros for missing frames (already initialized)
                
                # Apply transform if provided
                if self.transform:
                    seq_2d = self.transform(seq_2d)
                    seq_3d = self.transform(seq_3d)
                
                # Convert to torch tensors (only once at the end)
                return {
                    'keypoints_2d': torch.from_numpy(seq_2d), 
                    'keypoints_3d': torch.from_numpy(seq_3d), 
                    'frame_id': start_frame_id,
                    'file_idx': file_idx
                }
            else:  # '2d' or '3d'
                # Get the first frame to determine shape
                first_frame = self._load_frame_data(file_idx, frame_ids[0])
                keypoints_first = self._extract_keypoints_fast(first_frame)
                
                # Pre-allocate array for all frames
                seq = np.zeros((self.sequence_length, *keypoints_first.shape), dtype=np.float32)
                seq[0] = keypoints_first
                
                # Load subsequent frames
                for i in range(1, self.sequence_length):
                    try:
                        frame_data = self._load_frame_data(file_idx, frame_ids[i])
                        keypoints = self._extract_keypoints_fast(frame_data)
                        seq[i] = keypoints
                    except Exception as e:
                        logger.warning(f"Error loading sequence frame {frame_ids[i]}: {e}")
                        # Keep zeros for missing frames
                
                # Apply transform if provided
                if self.transform:
                    seq = self.transform(seq)
                
                # Convert to torch tensor (only once)
                return {
                    f'keypoints_{self.keypoint_type}': torch.from_numpy(seq), 
                    'frame_id': start_frame_id,
                    'file_idx': file_idx
                }
                
        except Exception as e:
            logger.error(f"Error loading sequence: {e}")
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
            # Handle single frame vs sequence differently
            if self.sequence_length > 1:
                # Use optimized sequence loading
                result = self._load_sequence_fast(file_idx, start_frame_id)
            else:  # Single frame
                # Load data
                data = self._load_frame_data(file_idx, start_frame_id)
                # Extract keypoints with optimized method
                keypoints = self._extract_keypoints_fast(data)
                
                # Apply transform if provided
                if self.transform:
                    if isinstance(keypoints, tuple):
                        keypoints = tuple(self.transform(k) for k in keypoints)
                    else:
                        keypoints = self.transform(keypoints)
                
                # Convert to torch tensor
                if isinstance(keypoints, tuple):
                    keypoints_2d, keypoints_3d = keypoints
                    
                    result = {
                        'keypoints_2d': torch.from_numpy(keypoints_2d).to(torch.float32), 
                        'keypoints_3d': torch.from_numpy(keypoints_3d).to(torch.float32), 
                        'frame_id': start_frame_id,
                        'file_idx': file_idx
                    }
                else:
                    result = {
                        f'keypoints_{self.keypoint_type}': torch.from_numpy(keypoints).to(torch.float32), 
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
                            logger.info(f"Cache hit rate: {hit_rate:.1f}%, Avg load time: {avg_load_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            # Return a dummy item if there's an error to avoid crashing the training
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
            'cache_size': len(self.frame_cache.cache),
            'max_cache_size': self.cache_size
        }
    
    def compute_dataset_stats(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Compute mean and standard deviation of the dataset.
        Returns ((mean_2d, std_2d), (mean_3d, std_3d))
        """
        logger.info("Computing dataset statistics...")
        
        # Use optimized calculation with parallel processing if available
        try:
            import concurrent.futures
            from functools import partial
            
            # Pre-allocate arrays for statistics
            joint_count = 133 if self.joint_indices is None else len(self.joint_indices)
            
            # Shared arrays for accumulating statistics
            sum_2d = np.zeros((joint_count, 2))
            sum_3d = np.zeros((joint_count, 3))
            sum_sq_2d = np.zeros((joint_count, 2))
            sum_sq_3d = np.zeros((joint_count, 3))
            count = 0
            
            # Sample a subset of the dataset for efficiency
            indices = np.random.choice(len(self), min(10000, len(self)), replace=False)
            
            # Use batched sampling for efficiency
            batch_size = 100
            for batch_idx in range(0, len(indices), batch_size):
                batch_indices = indices[batch_idx:batch_idx+batch_size]
                
                # Get all samples in current batch
                samples = [self[idx] for idx in batch_indices]
                
                for sample in samples:
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
        except Exception as e:
            logger.error(f"Error in compute_dataset_stats: {e}")
            raise

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