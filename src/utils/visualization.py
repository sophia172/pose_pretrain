"""
Visualization utilities for Human Pose Estimation.

This module provides utilities for visualizing 2D and 3D pose data.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Union, Dict


def plot_2d_pose(
    keypoints: Union[np.ndarray, torch.Tensor],
    connections: Optional[List[Tuple[int, int]]] = None,
    image: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 8),
    joint_color: str = 'blue',
    connection_color: str = 'red',
    joint_size: int = 8,
    line_width: int = 2,
    title: Optional[str] = None,
    axis_equal: bool = True,
    invert_y: bool = True,
    joint_labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot 2D pose keypoints.
    
    Args:
        keypoints: 2D keypoints of shape (num_joints, 2)
        connections: List of joint connections as pairs of indices
        image: Optional background image of shape (height, width, 3)
        figsize: Figure size
        joint_color: Color of the joints
        connection_color: Color of the connections
        joint_size: Size of the joints
        line_width: Width of the connection lines
        title: Title of the plot
        axis_equal: Whether to use equal aspect ratio
        invert_y: Whether to invert the y-axis (for image coordinates)
        joint_labels: Optional list of joint labels
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    
    # Create default joint connections if not provided
    if connections is None:
        # Default connections for a 17-joint human pose
        # This should be adapted to the specific skeleton structure
        connections = [
            (0, 1), (1, 2), (2, 3),  # Head and neck
            (1, 4), (4, 5), (5, 6),  # Right arm
            (1, 7), (7, 8), (8, 9),  # Left arm
            (0, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15)   # Left leg
        ]
    
    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    # Plot background image if provided
    if image is not None:
        ax.imshow(image)
    
    # Plot joint connections
    for connection in connections:
        joint1, joint2 = connection
        if joint1 < len(keypoints) and joint2 < len(keypoints):
            ax.plot(
                [keypoints[joint1, 0], keypoints[joint2, 0]],
                [keypoints[joint1, 1], keypoints[joint2, 1]],
                color=connection_color,
                linewidth=line_width
            )
    
    # Plot joints
    ax.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c=joint_color,
        s=joint_size,
        zorder=3  # Ensure joints are on top
    )
    
    # Add joint labels if provided
    if joint_labels is not None:
        for i, label in enumerate(joint_labels):
            if i < len(keypoints):
                ax.annotate(
                    label,
                    (keypoints[i, 0], keypoints[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center'
                )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    
    # Set equal aspect ratio
    if axis_equal:
        ax.set_aspect('equal')
    
    # Invert y-axis for image coordinates
    if invert_y:
        ax.invert_yaxis()
    
    # Remove axis ticks and labels if plotting on an image
    if image is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig


def plot_3d_pose(
    keypoints: Union[np.ndarray, torch.Tensor],
    connections: Optional[List[Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (10, 10),
    joint_color: str = 'blue',
    connection_color: str = 'red',
    joint_size: int = 50,
    line_width: int = 2,
    title: Optional[str] = None,
    axis_equal: bool = True,
    joint_labels: Optional[List[str]] = None,
    view_angles: Tuple[int, int] = (30, 45),
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot 3D pose keypoints.
    
    Args:
        keypoints: 3D keypoints of shape (num_joints, 3)
        connections: List of joint connections as pairs of indices
        figsize: Figure size
        joint_color: Color of the joints
        connection_color: Color of the connections
        joint_size: Size of the joints
        line_width: Width of the connection lines
        title: Title of the plot
        axis_equal: Whether to use equal aspect ratio
        joint_labels: Optional list of joint labels
        view_angles: (elevation, azimuth) viewing angles in degrees
        ax: Optional matplotlib 3D axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    
    # Create default joint connections if not provided
    if connections is None:
        # Default connections for a 17-joint human pose
        # This should be adapted to the specific skeleton structure
        connections = [
            (0, 1), (1, 2), (2, 3),  # Head and neck
            (1, 4), (4, 5), (5, 6),  # Right arm
            (1, 7), (7, 8), (8, 9),  # Left arm
            (0, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15)   # Left leg
        ]
    
    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot joint connections
    for connection in connections:
        joint1, joint2 = connection
        if joint1 < len(keypoints) and joint2 < len(keypoints):
            ax.plot(
                [keypoints[joint1, 0], keypoints[joint2, 0]],
                [keypoints[joint1, 1], keypoints[joint2, 1]],
                [keypoints[joint1, 2], keypoints[joint2, 2]],
                color=connection_color,
                linewidth=line_width
            )
    
    # Plot joints
    ax.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        c=joint_color,
        s=joint_size,
        zorder=3  # Ensure joints are on top
    )
    
    # Add joint labels if provided
    if joint_labels is not None:
        for i, label in enumerate(joint_labels):
            if i < len(keypoints):
                ax.text(
                    keypoints[i, 0],
                    keypoints[i, 1],
                    keypoints[i, 2],
                    label,
                    size=8,
                    zorder=4
                )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    
    # Set equal aspect ratio
    if axis_equal:
        # Get axis limits
        x_bounds = ax.get_xlim3d()
        y_bounds = ax.get_ylim3d()
        z_bounds = ax.get_zlim3d()
        
        # Find max range
        x_range = abs(x_bounds[1] - x_bounds[0])
        y_range = abs(y_bounds[1] - y_bounds[0])
        z_range = abs(z_bounds[1] - z_bounds[0])
        max_range = max(x_range, y_range, z_range)
        
        # Set equal aspect ratio
        mid_x = np.mean(x_bounds)
        mid_y = np.mean(y_bounds)
        mid_z = np.mean(z_bounds)
        
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angle
    elevation, azimuth = view_angles
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig


def plot_2d_3d_comparison(
    keypoints_2d: Union[np.ndarray, torch.Tensor],
    keypoints_3d: Union[np.ndarray, torch.Tensor],
    connections: Optional[List[Tuple[int, int]]] = None,
    image: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 8),
    title: Optional[str] = None,
    subtitles: Tuple[str, str] = ('2D Pose', '3D Pose'),
    view_angles: Tuple[int, int] = (30, 45)
) -> Figure:
    """
    Plot side-by-side comparison of 2D and 3D poses.
    
    Args:
        keypoints_2d: 2D keypoints of shape (num_joints, 2)
        keypoints_3d: 3D keypoints of shape (num_joints, 3)
        connections: List of joint connections as pairs of indices
        image: Optional background image for 2D pose
        figsize: Figure size
        title: Main title for the figure
        subtitles: Titles for the 2D and 3D subplots
        view_angles: (elevation, azimuth) viewing angles for 3D plot
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 2D pose subplot
    ax1 = fig.add_subplot(121)
    plot_2d_pose(
        keypoints=keypoints_2d,
        connections=connections,
        image=image,
        title=subtitles[0],
        ax=ax1
    )
    
    # 3D pose subplot
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_pose(
        keypoints=keypoints_3d,
        connections=connections,
        title=subtitles[1],
        view_angles=view_angles,
        ax=ax2
    )
    
    # Set main title
    if title is not None:
        fig.suptitle(title, fontsize=14)
        
    # Ensure tight layout
    plt.tight_layout()
    
    return fig


def plot_pose_sequence(
    keypoints: Union[np.ndarray, torch.Tensor],
    connections: Optional[List[Tuple[int, int]]] = None,
    is_3d: bool = False,
    num_frames: int = 5,
    figsize: Tuple[int, int] = (20, 4),
    title: Optional[str] = None
) -> Figure:
    """
    Plot a sequence of poses.
    
    Args:
        keypoints: Pose keypoints of shape (seq_len, num_joints, 2|3)
        connections: List of joint connections as pairs of indices
        is_3d: Whether the poses are 3D
        num_frames: Number of frames to plot
        figsize: Figure size
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    
    # Determine number of frames to plot
    seq_len = len(keypoints)
    num_frames = min(num_frames, seq_len)
    
    # Select frames with equal spacing
    if seq_len <= num_frames:
        frame_indices = list(range(seq_len))
    else:
        frame_indices = [int(i * (seq_len - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot each frame
    for i, idx in enumerate(frame_indices):
        if is_3d:
            ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
            plot_3d_pose(
                keypoints=keypoints[idx],
                connections=connections,
                title=f"Frame {idx}",
                ax=ax
            )
        else:
            ax = fig.add_subplot(1, num_frames, i + 1)
            plot_2d_pose(
                keypoints=keypoints[idx],
                connections=connections,
                title=f"Frame {idx}",
                ax=ax
            )
    
    # Set main title
    if title is not None:
        fig.suptitle(title, fontsize=14)
        
    # Ensure tight layout
    plt.tight_layout()
    
    return fig


def plot_error_heatmap(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    joint_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Joint Position Error Heatmap',
    cmap: str = 'viridis'
) -> Figure:
    """
    Plot a heatmap of per-joint errors.
    
    Args:
        predictions: Predicted 3D poses of shape (batch_size, num_joints, 3)
        targets: Target 3D poses of shape (batch_size, num_joints, 3)
        joint_names: Names of the joints
        figsize: Figure size
        title: Title for the figure
        cmap: Colormap for the heatmap
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Calculate per-joint errors
    joint_errors = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))  # (batch_size, num_joints)
    
    # Calculate mean and std of errors per joint
    mean_errors = np.mean(joint_errors, axis=0)  # (num_joints,)
    std_errors = np.std(joint_errors, axis=0)  # (num_joints,)
    
    # Create joint names if not provided
    num_joints = mean_errors.shape[0]
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(num_joints)]
    else:
        # Ensure joint_names matches the number of joints
        if len(joint_names) != num_joints:
            joint_names = joint_names[:num_joints] if len(joint_names) > num_joints else joint_names + [f"Joint {i}" for i in range(len(joint_names), num_joints)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(mean_errors.reshape(1, -1), cmap=cmap)
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks(np.arange(num_joints))
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean Error (mm)', rotation=-90, va="bottom")
    
    # Add error values as text
    for i in range(num_joints):
        ax.text(i, 0, f"{mean_errors[i]:.1f}\n±{std_errors[i]:.1f}",
                ha="center", va="center", color="white" if mean_errors[i] > np.mean(mean_errors) else "black")
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig


def save_visualization(
    fig: Figure,
    filename: str,
    output_dir: str = 'outputs/visualizations',
    dpi: int = 200,
    close_figure: bool = True
) -> str:
    """
    Save visualization to file.
    
    Args:
        fig: Matplotlib figure
        filename: Filename to save as
        output_dir: Directory to save to
        dpi: DPI for the saved image
        close_figure: Whether to close the figure after saving
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename has an extension
    if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
        filename += '.png'
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    # Close figure if requested
    if close_figure:
        plt.close(fig)
    
    return output_path


def visualize_sample_predictions(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    indices: Optional[List[int]] = None,
    num_samples: int = 5,
    connections: Optional[List[Tuple[int, int]]] = None,
    output_dir: str = 'outputs/visualizations',
    prefix: str = 'sample'
) -> List[str]:
    """
    Visualize sample predictions and save them to files.
    
    Args:
        predictions: Predicted 3D poses of shape (batch_size, num_joints, 3)
        targets: Target 3D poses of shape (batch_size, num_joints, 3)
        indices: Indices of samples to visualize (if None, random samples are chosen)
        num_samples: Number of samples to visualize (if indices is None)
        connections: List of joint connections as pairs of indices
        output_dir: Directory to save visualizations to
        prefix: Prefix for filenames
        
    Returns:
        List of paths to saved visualization files
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples to visualize
    batch_size = len(predictions)
    if indices is None:
        if num_samples >= batch_size:
            indices = list(range(batch_size))
        else:
            indices = np.random.choice(batch_size, num_samples, replace=False)
    
    # Visualize each sample
    output_paths = []
    for i, idx in enumerate(indices):
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        
        # Plot target pose
        ax1 = fig.add_subplot(121, projection='3d')
        plot_3d_pose(
            keypoints=targets[idx],
            connections=connections,
            title='Ground Truth',
            ax=ax1
        )
        
        # Plot predicted pose
        ax2 = fig.add_subplot(122, projection='3d')
        plot_3d_pose(
            keypoints=predictions[idx],
            connections=connections,
            title='Prediction',
            ax=ax2
        )
        
        # Calculate error
        error = np.mean(np.sqrt(np.sum((predictions[idx] - targets[idx]) ** 2, axis=-1)))
        fig.suptitle(f'Sample {idx} - Mean Error: {error:.2f} mm', fontsize=14)
        
        # Save figure
        output_path = os.path.join(output_dir, f'{prefix}_{i}.png')
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        output_paths.append(output_path)
    
    return output_paths 