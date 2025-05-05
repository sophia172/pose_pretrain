import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Union

class Human36MDataset(Dataset):
    """
    PyTorch Dataset for Human3.6M dataset that loads from JSON files containing 2D and 3D keypoints
    """
    def __init__(
        self, 
        json_files: List[str],
        keypoint_type: str = 'both',  # 'both', '2d', or '3d'
        joint_indices: Optional[List[int]] = None,
        transform=None,
        preload: bool = False,
        sequence_length: int = 1,
        stride: int = 1
    ):
        """
        Args:
            json_files: List of paths to JSON files
            keypoint_type: Type of keypoints to return ('both', '2d', or '3d')
            joint_indices: List of joint indices to include. If None, include all joints
            transform: Optional transform to apply to the data
            preload: If True, load all data into memory. If False, load on-the-fly
            sequence_length: Number of consecutive frames to return (for sequence data)
            stride: Stride between consecutive sequences
        """
        self.json_files = json_files
        self.keypoint_type = keypoint_type
        self.joint_indices = joint_indices
        self.transform = transform
        self.preload = preload
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Map frame IDs to (file_idx, frame_idx) for retrieval
        self.frame_mapping = []
        self.data_cache = {}
        
        print(f"Initializing dataset with {len(json_files)} JSON files")
        
        # Scan files to build frame mapping
        for file_idx, json_file in enumerate(json_files):
            try:
                print(f"Scanning {os.path.basename(json_file)}...")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process all frame IDs in ascending order
                frame_ids = sorted([int(k) for k in data.keys()])
                
                for frame_idx, frame_id in enumerate(frame_ids):
                    # For sequences, ensure we have enough frames after this one
                    if frame_idx + self.sequence_length <= len(frame_ids):
                        self.frame_mapping.append((file_idx, str(frame_id)))
                
                # Preload data if required
                if self.preload:
                    self.data_cache[file_idx] = data
                    print(f"Preloaded file {file_idx} with {len(data)} frames")
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Apply stride to frame mapping (take every nth sequence)
        self.frame_mapping = self.frame_mapping[::self.stride]
        
        print(f"Dataset initialized with {len(self.frame_mapping)} sequences")
        
    def __len__(self):
        return len(self.frame_mapping)
    
    def _load_frame_data(self, file_idx: int, frame_id: str) -> dict:
        """Load data for a specific frame"""
        if self.preload and file_idx in self.data_cache:
            return self.data_cache[file_idx][frame_id]
        
        # Load data from file
        with open(self.json_files[file_idx], 'r') as f:
            data = json.load(f)
        
        return data[frame_id]
    
    def _extract_keypoints(self, frame_data: dict) -> tuple:
        """Extract 2D and/or 3D keypoints from frame data"""
        # Initialize empty arrays
        keypoints_2d = []
        keypoints_3d = []
        
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
                        # Handle missing joints
                        keypoints_2d.append([0.0, 0.0])
                
                keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
            else:
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
                        # Handle missing joints
                        keypoints_3d.append([0.0, 0.0, 0.0])
                
                keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
            else:
                raise KeyError("3D keypoints not found in data")
        
        # Return the appropriate keypoints based on keypoint_type
        if self.keypoint_type == '2d':
            return keypoints_2d
        elif self.keypoint_type == '3d':
            return keypoints_3d
        else:  # 'both'
            return keypoints_2d, keypoints_3d
    
    def __getitem__(self, idx):
        """Get a single item or sequence from the dataset"""
        # Get the mapping for this index
        file_idx, start_frame_id = self.frame_mapping[idx]
        
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
                if self.sequence_length > 1:
                    # Load subsequent frames
                    # This assumes frame IDs are consecutive integers
                    for i in range(1, self.sequence_length):
                        next_frame_id = str(int(start_frame_id) + i)
                        
                        try:
                            data = self._load_frame_data(file_idx, next_frame_id)
                            keypoints_2d, keypoints_3d = self._extract_keypoints(data)
                            
                            seq_2d[i] = keypoints_2d
                            seq_3d[i] = keypoints_3d
                        except Exception as e:
                            print(f"Error loading sequence frame {next_frame_id}: {e}")
                            # Keep zeros for missing frames
                
                # Apply transform if provided
                if self.transform:
                    seq_2d = self.transform(seq_2d)
                    seq_3d = self.transform(seq_3d)
                
                # Convert to torch tensors
                seq_2d = torch.from_numpy(seq_2d)
                seq_3d = torch.from_numpy(seq_3d)
                
                return {'keypoints_2d': seq_2d, 'keypoints_3d': seq_3d, 'frame_id': start_frame_id}
                
            else:  # '2d' or '3d'
                # Extract keypoints for the first frame to determine shape
                keypoints_first = self._extract_keypoints(data)
                
                seq = np.zeros((self.sequence_length, *keypoints_first.shape), dtype=np.float32)
                seq[0] = keypoints_first
                
                # Get consecutive frames if needed
                if self.sequence_length > 1:
                    for i in range(1, self.sequence_length):
                        next_frame_id = str(int(start_frame_id) + i)
                        
                        try:
                            data = self._load_frame_data(file_idx, next_frame_id)
                            keypoints = self._extract_keypoints(data)
                            seq[i] = keypoints
                        except Exception as e:
                            print(f"Error loading sequence frame {next_frame_id}: {e}")
                            # Keep zeros for missing frames
                
                # Apply transform if provided
                if self.transform:
                    seq = self.transform(seq)
                
                # Convert to torch tensor
                seq = torch.from_numpy(seq)
                
                return {f'keypoints_{self.keypoint_type}': seq, 'frame_id': start_frame_id}
                
        else:  # Single frame
            # Load data
            data = self._load_frame_data(file_idx, start_frame_id)
            
            # Extract keypoints
            result = self._extract_keypoints(data)
            
            # Apply transform if provided
            if self.transform:
                if isinstance(result, tuple):
                    result = tuple(self.transform(r) for r in result)
                else:
                    result = self.transform(result)
            
            # Convert to torch tensor
            if isinstance(result, tuple):
                result = tuple(torch.from_numpy(r) for r in result)
                return {'keypoints_2d': result[0], 'keypoints_3d': result[1], 'frame_id': start_frame_id}
            else:
                result = torch.from_numpy(result)
                return {f'keypoints_{self.keypoint_type}': result, 'frame_id': start_frame_id}

# Helper function to get dataloader
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
    drop_last: bool = False
):
    """
    Create a DataLoader for the Human3.6M dataset
    
    Args:
        json_files: List of paths to JSON files
        batch_size: Batch size for the DataLoader
        keypoint_type: Type of keypoints to return ('both', '2d', or '3d')
        joint_indices: List of joint indices to include. If None, include all joints
        transform: Optional transform to apply to the data
        preload: If True, load all data into memory. If False, load on-the-fly
        sequence_length: Number of consecutive frames to return (for sequence data)
        stride: Stride between consecutive sequences
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last incomplete batch
    
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
        pin_memory=True  # Speeds up data transfer to GPU
    )
    
    return dataloader

# Example usage
if __name__ == "__main__":
    # Define paths to all the split JSON files
    
    json_files = [
        train_part1, 
        train_part2, 
        train_part3, 
        train_part4, 
        train_part5
    ]
    
    # Create a dataloader for training
    train_loader = get_human36m_dataloader(
        json_files=json_files,
        batch_size=16,
        keypoint_type='both',  # Get both 2D and 3D keypoints
        joint_indices=None,  # Include all joints
        preload=False,  # Load on-the-fly to save memory
        sequence_length=1,  # Single frames
        shuffle=True,
        num_workers=4
    )
    
    # Demonstrate usage - get a batch and print shapes
    for i, batch in enumerate(train_loader):
        if 'keypoints_2d' in batch:
            print(f"2D keypoints shape: {batch['keypoints_2d'].shape}")
        if 'keypoints_3d' in batch:
            print(f"3D keypoints shape: {batch['keypoints_3d'].shape}")
        print(f"Frame IDs: {batch['frame_id']}")
        
        # Only show first batch
        if i == 0:
            break

    # Example with sequences
    seq_loader = get_human36m_dataloader(
        json_files=json_files,
        batch_size=8,
        keypoint_type='both',
        sequence_length=10,  # Get 10 consecutive frames
        stride=5,  # Sample sequences with stride 5
        shuffle=True
    )
    
    # Print sequence shapes
    for i, batch in enumerate(seq_loader):
        if 'keypoints_2d' in batch:
            print(f"2D sequence shape: {batch['keypoints_2d'].shape}")
        if 'keypoints_3d' in batch:
            print(f"3D sequence shape: {batch['keypoints_3d'].shape}")
        print(f"Starting Frame IDs: {batch['frame_id']}")
        
        # Only show first batch
        if i == 0:
            break