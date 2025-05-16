"""
Loss functions for Human Pose Estimation models.

This module contains various loss functions used for training
pose estimation models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for 3D pose estimation.
    
    Implements various types of reconstruction losses for comparing
    predicted and target 3D poses.
    """
    
    def __init__(self, loss_type: str = "l1", reduction: str = "mean", epsilon: float = 1e-8):
        """
        Initialize reconstruction loss.
        
        Args:
            loss_type: Type of loss ("l1", "mse", "huber", "wing")
            reduction: Reduction method ("mean", "sum", "none")
            epsilon: Small value to prevent numerical instability
        """
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.epsilon = epsilon
        
        # Setup loss function based on type
        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        elif self.loss_type == "wing":
            # Wing loss parameters
            self.wing_w = 10.0
            self.wing_epsilon = max(2.0, epsilon)  # Ensure positive value
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred: Predicted 3D poses of shape (batch_size, num_joints, 3) or (batch_size, seq_len, num_joints, 3)
            target: Target 3D poses of same shape as pred
            mask: Optional mask for valid joints/frames
            
        Returns:
            Loss tensor
        """
        if self.loss_type in ["l1", "mse", "huber"]:
            if mask is not None:
                # Apply mask if provided
                if self.reduction == "none":
                    loss = self.loss_fn(pred, target)
                    return loss * mask
                else:
                    # For masked mean/sum, we need to handle reduction manually
                    loss = F.l1_loss(pred, target, reduction="none") if self.loss_type == "l1" else \
                           F.mse_loss(pred, target, reduction="none") if self.loss_type == "mse" else \
                           F.smooth_l1_loss(pred, target, reduction="none")
                    
                    # Apply mask and reduce
                    loss = loss * mask
                    mask_sum = mask.sum().clamp(min=self.epsilon)  # Prevent division by zero
                    
                    if self.reduction == "mean":
                        return loss.sum() / mask_sum
                    else:  # sum
                        return loss.sum()
            else:
                # No mask, use the loss function directly
                return self.loss_fn(pred, target)
        elif self.loss_type == "wing":
            # Wing loss implementation
            abs_diff = (pred - target).abs()
            
            # Apply wing loss formula with safety checks
            c = self.wing_w * (1.0 - torch.log(1.0 + self.wing_w / self.wing_epsilon))
            
            # Safe logarithm calculation
            safe_input = torch.clamp(abs_diff / self.wing_epsilon, min=0, max=1e6)
            
            # Calculate loss using the wing loss formula with safety checks
            wing_loss = torch.where(
                abs_diff < self.wing_w,
                self.wing_w * torch.log1p(safe_input),  # Using log1p for better stability
                abs_diff - c
            )
            
            # Apply mask if provided
            if mask is not None:
                wing_loss = wing_loss * mask
                mask_sum = mask.sum().clamp(min=self.epsilon)  # Prevent division by zero
                
            # Apply reduction
            if self.reduction == "mean":
                if mask is not None:
                    return wing_loss.sum() / mask_sum
                else:
                    return wing_loss.mean()
            elif self.reduction == "sum":
                return wing_loss.sum()
            else:  # none
                return wing_loss
        
        # Should not reach here
        raise ValueError(f"Unsupported loss type: {self.loss_type}")


class LimbConsistencyLoss(nn.Module):
    """
    Limb consistency loss for ensuring anatomical plausibility.
    
    Encourages the model to maintain consistent limb lengths and
    proportions in the predicted poses.
    """
    
    def __init__(
        self, 
        limb_connections: Optional[List[Tuple[int, int]]] = None,
        loss_type: str = "l1",
        reduction: str = "mean",
        bidirectional: bool = True,
        epsilon: float = 1e-8
    ):
        """
        Initialize limb consistency loss.
        
        Args:
            limb_connections: List of joint index pairs defining limbs
            loss_type: Type of loss ("l1", "mse", "huber")
            reduction: Reduction method ("mean", "sum", "none")
            bidirectional: Whether to compute loss in both directions (pred->target and target->pred)
            epsilon: Small value for numerical stability
        """
        super().__init__()
        
        # Default limb connections if not provided
        if limb_connections is None:
            # Example limb connections for a standard human skeleton
            # Should be adapted to the specific joint configuration
            self.limb_connections = [
                (0, 1), (1, 2), (2, 3),  # Head, neck, spine
                (1, 4), (4, 5), (5, 6),  # Right arm
                (1, 7), (7, 8), (8, 9),  # Left arm
                (0, 10), (10, 11), (11, 12),  # Right leg
                (0, 13), (13, 14), (14, 15)   # Left leg
            ]
        else:
            self.limb_connections = limb_connections
            
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        
        # Setup loss function
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute limb consistency loss.
        
        Args:
            pred: Predicted 3D poses of shape (batch_size, num_joints, 3) or (batch_size, seq_len, num_joints, 3)
            target: Target 3D poses of same shape as pred
            
        Returns:
            Loss tensor
        """
        # Calculate limb lengths
        pred_lengths = self._compute_limb_lengths(pred)
        target_lengths = self._compute_limb_lengths(target)
        
        # Calculate limb length loss
        if self.bidirectional:
            # Bidirectional loss: penalize both over- and under-estimation equally
            forward_loss = self.loss_fn(pred_lengths, target_lengths, reduction=self.reduction)
            backward_loss = self.loss_fn(target_lengths, pred_lengths, reduction=self.reduction)
            return (forward_loss + backward_loss) / 2
        else:
            # Unidirectional loss
            return self.loss_fn(pred_lengths, target_lengths, reduction=self.reduction)
            
    def _compute_limb_lengths(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute limb lengths for all poses in the batch.
        
        Args:
            poses: 3D poses of shape (batch_size, num_joints, 3) or (batch_size, seq_len, num_joints, 3)
            
        Returns:
            Tensor of limb lengths
        """
        is_sequence = len(poses.shape) == 4
        
        # Calculate limb lengths for each limb connection
        limb_lengths = []
        
        for joint1, joint2 in self.limb_connections:
            if is_sequence:
                # Sequence data: (batch_size, seq_len, num_joints, 3)
                limb_vec = poses[:, :, joint1, :] - poses[:, :, joint2, :]
                length = torch.sqrt(torch.sum(limb_vec * limb_vec, dim=-1) + self.epsilon)  # Stable norm calculation
            else:
                # Single frame data: (batch_size, num_joints, 3)
                limb_vec = poses[:, joint1, :] - poses[:, joint2, :]
                length = torch.sqrt(torch.sum(limb_vec * limb_vec, dim=-1) + self.epsilon)  # Stable norm calculation
                
            limb_lengths.append(length)
            
        # Stack along the last dimension
        return torch.stack(limb_lengths, dim=-1)


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal smoothness loss for sequential pose data.
    
    Penalizes sudden changes in position, velocity, or acceleration
    to encourage temporally consistent predictions.
    """
    
    def __init__(
        self, 
        loss_type: str = "l1",
        reduction: str = "mean",
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        epsilon: float = 1e-8,
        max_value: float = 1e6  # Maximum value to clip extreme values
    ):
        """
        Initialize temporal smoothness loss.
        
        Args:
            loss_type: Type of loss ("l1", "mse", "huber")
            reduction: Reduction method ("mean", "sum", "none")
            velocity_weight: Weight for velocity smoothness term
            acceleration_weight: Weight for acceleration smoothness term
            epsilon: Small value to prevent numerical instability
            max_value: Maximum value to clip extremes
        """
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.epsilon = epsilon
        self.max_value = max_value
        
        # Setup loss function
        if self.loss_type == "l1":
            self.loss_fn = lambda x: torch.abs(x)
        elif self.loss_type == "mse":
            self.loss_fn = lambda x: torch.square(x)
        elif self.loss_type == "huber":
            self.beta = 1.0  # Huber beta parameter
            self.loss_fn = lambda x: torch.where(
                x.abs() < self.beta,
                0.5 * torch.square(x),
                self.beta * (x.abs() - 0.5 * self.beta)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def forward(self, poses: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal smoothness loss.
        
        Args:
            poses: Sequential 3D poses of shape (batch_size, seq_len, num_joints, 3)
            mask: Optional mask for valid frames
            
        Returns:
            Loss tensor
        """
        batch_size, seq_len, num_joints, _ = poses.shape
        
        # Require at least 3 frames for acceleration
        if seq_len < 3:
            return torch.tensor(0.0, device=poses.device)
        
        # Clip poses to prevent extreme values    
        poses_clipped = torch.clamp(poses, -self.max_value, self.max_value)
            
        # Calculate velocities (first derivatives)
        velocities = poses_clipped[:, 1:] - poses_clipped[:, :-1]  # (batch_size, seq_len-1, num_joints, 3)
        
        # Calculate accelerations (second derivatives)
        accelerations = velocities[:, 1:] - velocities[:, :-1]  # (batch_size, seq_len-2, num_joints, 3)
        
        # Apply loss function to velocity and acceleration
        velocity_loss = self.loss_fn(velocities)
        acceleration_loss = self.loss_fn(accelerations)
        
        # Apply mask if provided
        if mask is not None:
            # Adjust mask dimensions for velocity and acceleration
            velocity_mask = mask[:, 1:] * mask[:, :-1]
            acceleration_mask = velocity_mask[:, 1:] * velocity_mask[:, :-1]
            
            velocity_loss = velocity_loss * velocity_mask.unsqueeze(-1).unsqueeze(-1)
            acceleration_loss = acceleration_loss * acceleration_mask.unsqueeze(-1).unsqueeze(-1)
            
            # Count number of valid entries for mean reduction
            valid_velocity_entries = velocity_mask.sum().clamp(min=self.epsilon)
            valid_acceleration_entries = acceleration_mask.sum().clamp(min=self.epsilon)
        else:
            # All entries are valid
            valid_velocity_entries = batch_size * (seq_len - 1) * num_joints
            valid_acceleration_entries = batch_size * (seq_len - 2) * num_joints
            
        # Compute mean or sum based on reduction type
        if self.reduction == "mean":
            velocity_term = velocity_loss.sum() / valid_velocity_entries
            acceleration_term = acceleration_loss.sum() / valid_acceleration_entries
        elif self.reduction == "sum":
            velocity_term = velocity_loss.sum()
            acceleration_term = acceleration_loss.sum()
        else:  # "none"
            return self.velocity_weight * velocity_loss, self.acceleration_weight * acceleration_loss
            
        # Combine weighted terms
        return self.velocity_weight * velocity_term + self.acceleration_weight * acceleration_term


class JointAngleLoss(nn.Module):
    """
    Joint angle loss for encouraging anatomically plausible joint angles.
    
    Penalizes implausible joint angles based on angle limits of human joints.
    """
    
    def __init__(
        self,
        joint_chains: List[Tuple[int, int, int]],
        angle_limits: Optional[List[Tuple[float, float]]] = None,
        loss_type: str = "l1",
        reduction: str = "mean",
        epsilon: float = 1e-8
    ):
        """
        Initialize joint angle loss.
        
        Args:
            joint_chains: List of joint triplets (parent, joint, child) for angle calculation
            angle_limits: Optional list of (min_angle, max_angle) in radians for each joint chain
            loss_type: Type of loss ("l1", "mse", "huber")
            reduction: Reduction method ("mean", "sum", "none")
            epsilon: Small value to prevent numerical instability
        """
        super().__init__()
        
        self.joint_chains = joint_chains
        self.angle_limits = angle_limits
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.epsilon = epsilon
        
        # If no angle limits provided, use default range (0, π)
        if self.angle_limits is None:
            self.angle_limits = [(0.0, 3.14159)] * len(joint_chains)
            
        assert len(self.joint_chains) == len(self.angle_limits), \
            "Number of joint chains must match number of angle limits"
            
        # Setup loss function
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute joint angle loss.
        
        Args:
            poses: 3D poses of shape (batch_size, num_joints, 3) or (batch_size, seq_len, num_joints, 3)
            
        Returns:
            Loss tensor
        """
        is_sequence = len(poses.shape) == 4
        
        if is_sequence:
            batch_size, seq_len, num_joints, _ = poses.shape
            # Reshape to (batch_size * seq_len, num_joints, 3)
            poses_flat = poses.reshape(-1, num_joints, 3)
        else:
            batch_size, num_joints, _ = poses.shape
            poses_flat = poses
            
        # Calculate angles for each joint chain
        angles = []
        angle_limits_tensor = []
        
        for i, (parent, joint, child) in enumerate(self.joint_chains):
            # Get vectors for both limbs in the chain
            v1 = poses_flat[:, parent, :] - poses_flat[:, joint, :]  # parent -> joint
            v2 = poses_flat[:, child, :] - poses_flat[:, joint, :]   # joint -> child
            
            # Check for zero vectors and add epsilon to prevent NaN in normalization
            v1_norm_sq = torch.sum(v1 * v1, dim=-1, keepdim=True)
            v2_norm_sq = torch.sum(v2 * v2, dim=-1, keepdim=True)
            
            # Skip normalization for zero vectors
            v1_is_zero = v1_norm_sq < self.epsilon
            v2_is_zero = v2_norm_sq < self.epsilon
            
            # Replace zero vectors with unit vectors to avoid NaN
            safe_v1 = torch.where(v1_is_zero, torch.tensor([1.0, 0.0, 0.0], device=poses.device), v1)
            safe_v2 = torch.where(v2_is_zero, torch.tensor([0.0, 1.0, 0.0], device=poses.device), v2)
            
            # Normalize vectors with epsilon for numerical stability
            v1_norm = safe_v1 / torch.sqrt(torch.sum(safe_v1 * safe_v1, dim=-1, keepdim=True) + self.epsilon)
            v2_norm = safe_v2 / torch.sqrt(torch.sum(safe_v2 * safe_v2, dim=-1, keepdim=True) + self.epsilon)
            
            # Calculate angle using cross product (more stable)
            cross_prod = torch.cross(v1_norm, v2_norm, dim=-1)
            cross_norm = torch.norm(cross_prod, dim=-1)
            dot_prod = torch.sum(v1_norm * v2_norm, dim=-1)
            
            # Clamp dot product to valid range for numerical stability
            dot_prod_clamped = torch.clamp(dot_prod, -1.0 + self.epsilon, 1.0 - self.epsilon)
            
            # Calculate angle, handle cases where both vectors are zero
            valid_angle = torch.atan2(cross_norm, dot_prod_clamped)
            angle = torch.where(
                v1_is_zero.squeeze(-1) | v2_is_zero.squeeze(-1),
                torch.zeros_like(valid_angle),  # Zero angle for zero vectors
                valid_angle
            )
            
            angles.append(angle)
            angle_limits_tensor.append(torch.tensor([self.angle_limits[i][0], self.angle_limits[i][1]], 
                                                   device=poses.device))
            
        # Stack angles and reshape
        angles = torch.stack(angles, dim=-1)  # (batch_size * seq_len, num_joint_chains) or (batch_size, num_joint_chains)
        angle_limits_tensor = torch.stack(angle_limits_tensor, dim=0)  # (num_joint_chains, 2)
        
        # Reshape back to original batch shape if needed
        if is_sequence:
            angles = angles.reshape(batch_size, seq_len, -1)
            
        # Calculate loss based on angle limits
        min_angles = angle_limits_tensor[:, 0].unsqueeze(0)  # (1, num_joint_chains)
        max_angles = angle_limits_tensor[:, 1].unsqueeze(0)  # (1, num_joint_chains)
        
        # Calculate how much angles are outside permitted range
        angle_errors = torch.zeros_like(angles)
        
        # Angles smaller than minimum
        below_min = angles < min_angles
        angle_errors = torch.where(below_min, min_angles - angles, angle_errors)
        
        # Angles larger than maximum
        above_max = angles > max_angles
        angle_errors = torch.where(above_max, angles - max_angles, angle_errors)
        
        # Apply loss function
        if self.reduction == "none":
            return self.loss_fn(angle_errors, torch.zeros_like(angle_errors), reduction="none")
        else:
            return self.loss_fn(angle_errors, torch.zeros_like(angle_errors), reduction=self.reduction)


class TotalLoss(nn.Module):
    """
    Combined loss function for pose estimation.
    
    Combines multiple loss terms with weights for overall training objective.
    """
    
    def __init__(
        self,
        reconstruction_loss: Optional[nn.Module] = None,
        consistency_loss: Optional[nn.Module] = None,
        smoothness_loss: Optional[nn.Module] = None,
        joint_angle_loss: Optional[nn.Module] = None,
        recon_weight: float = 1.0,
        consistency_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        angle_weight: float = 0.2,
        max_loss: float = 1e6  # Maximum loss value to prevent explosion
    ):
        """
        Initialize total loss function.
        
        Args:
            reconstruction_loss: Loss module for reconstruction
            consistency_loss: Loss module for limb consistency
            smoothness_loss: Loss module for temporal smoothness
            joint_angle_loss: Loss module for joint angles
            recon_weight: Weight for reconstruction loss
            consistency_weight: Weight for consistency loss
            smoothness_weight: Weight for smoothness loss
            angle_weight: Weight for joint angle loss
            max_loss: Maximum allowable loss value
        """
        super().__init__()
        
        # Setup loss components with defaults if not provided
        self.reconstruction_loss = reconstruction_loss or ReconstructionLoss(loss_type="l1")
        self.consistency_loss = consistency_loss
        self.smoothness_loss = smoothness_loss
        self.joint_angle_loss = joint_angle_loss
        
        # Loss weights
        self.recon_weight = recon_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.angle_weight = angle_weight
        self.max_loss = max_loss
        
        # For storing loss components
        self.last_loss_components = {}
        
        # Build human-readable description of the combined loss
        desc_parts = [f"{recon_weight:.2f}*{self.reconstruction_loss.__class__.__name__}"]
        
        if self.consistency_loss is not None and consistency_weight > 0:
            desc_parts.append(f"{consistency_weight:.2f}*{self.consistency_loss.__class__.__name__}")
            
        if self.smoothness_loss is not None and smoothness_weight > 0:
            desc_parts.append(f"{smoothness_weight:.2f}*{self.smoothness_loss.__class__.__name__}")
            
        if self.joint_angle_loss is not None and angle_weight > 0:
            desc_parts.append(f"{angle_weight:.2f}*{self.joint_angle_loss.__class__.__name__}")
            
        self.description = " + ".join(desc_parts)
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total loss.
        
        Args:
            pred: Predicted 3D poses
            target: Target 3D poses
            mask: Optional mask for valid joints/frames
            
        Returns:
            Total loss value
        """
        # Detect and replace NaN values early
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=self.max_loss, neginf=-self.max_loss)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            target = torch.nan_to_num(target, nan=0.0, posinf=self.max_loss, neginf=-self.max_loss)
            
        if mask is not None and (torch.isnan(mask).any() or torch.isinf(mask).any()):
            mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        
        loss_components = {}
        
        # Reconstruction loss with safety
        try:
            recon_loss = self.reconstruction_loss(pred, target, mask)
            # Clip extremely large values
            recon_loss = torch.clamp(recon_loss, max=self.max_loss)
            loss_components["reconstruction"] = recon_loss
            total_loss = self.recon_weight * recon_loss
        except Exception as e:
            # Fall back to L1 if the main loss fails
            print(f"Warning: Reconstruction loss failed: {e}. Falling back to L1.")
            fallback_loss = F.l1_loss(pred, target, reduction="mean")
            loss_components["reconstruction"] = fallback_loss
            total_loss = self.recon_weight * fallback_loss
            
        # Limb consistency loss
        if self.consistency_loss is not None and self.consistency_weight > 0:
            try:
                consist_loss = self.consistency_loss(pred, target)
                consist_loss = torch.clamp(consist_loss, max=self.max_loss)
                loss_components["consistency"] = consist_loss
                total_loss = total_loss + self.consistency_weight * consist_loss
            except Exception as e:
                print(f"Warning: Consistency loss failed: {e}. Skipping this term.")
                loss_components["consistency"] = torch.tensor(0.0, device=pred.device)
            
        # Temporal smoothness loss (only for sequential data)
        is_sequence = len(pred.shape) == 4
        if is_sequence and self.smoothness_loss is not None and self.smoothness_weight > 0:
            try:
                smooth_loss = self.smoothness_loss(pred, mask)
                smooth_loss = torch.clamp(smooth_loss, max=self.max_loss)
                loss_components["smoothness"] = smooth_loss
                total_loss = total_loss + self.smoothness_weight * smooth_loss
            except Exception as e:
                print(f"Warning: Smoothness loss failed: {e}. Skipping this term.")
                loss_components["smoothness"] = torch.tensor(0.0, device=pred.device)
            
        # Joint angle loss
        if self.joint_angle_loss is not None and self.angle_weight > 0:
            try:
                angle_loss = self.joint_angle_loss(pred)
                angle_loss = torch.clamp(angle_loss, max=self.max_loss)
                loss_components["joint_angle"] = angle_loss
                total_loss = total_loss + self.angle_weight * angle_loss
            except Exception as e:
                print(f"Warning: Joint angle loss failed: {e}. Skipping this term.")
                loss_components["joint_angle"] = torch.tensor(0.0, device=pred.device)
                
        # Final check to prevent NaN from propagating
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf detected in final loss. Using fallback L1 loss.")
            total_loss = F.l1_loss(pred, target, reduction="mean")
            
        # Ensure the loss is bounded
        total_loss = torch.clamp(total_loss, max=self.max_loss)
            
        loss_components["total"] = total_loss
        
        # Store the loss components for later retrieval
        self.last_loss_components = loss_components
        
        return total_loss
    
    def get_loss_components(self) -> Dict[str, torch.Tensor]:
        """
        Get the components of the last computed loss.
        
        Returns:
            Dictionary containing individual loss components
        """
        return self.last_loss_components


def get_loss_fn(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Configuration dictionary with loss settings
        
    Returns:
        Loss module
    """
    training_config = config.get("training", {})
    losses_config = training_config.get("losses", {})
    
    # Get reconstruction loss settings
    reconstruction_config = losses_config.get("reconstruction", {})
    loss_type = reconstruction_config.get("type", "l1")
    epsilon = reconstruction_config.get("epsilon", 1e-8)
    
    # Extract weights with defaults
    recon_weight = reconstruction_config.get("weight", 1.0)
    
    # Get other loss weights
    consistency_config = losses_config.get("consistency", {})
    consistency_weight = consistency_config.get("weight", 0.5)
    
    smoothness_config = losses_config.get("smoothness", {})
    smoothness_weight = smoothness_config.get("weight", 0.1)
    
    # Joint angle loss is optional
    angle_config = losses_config.get("joint_angle", {})
    angle_weight = angle_config.get("weight", 0.0)
    
    # Create individual loss components
    reconstruction_loss = ReconstructionLoss(loss_type=loss_type, epsilon=epsilon)
    
    # Limb consistency loss (if weight > 0)
    consistency_loss = None
    if consistency_weight > 0:
        consistency_loss = LimbConsistencyLoss(
            loss_type=consistency_config.get("type", loss_type),
            epsilon=consistency_config.get("epsilon", epsilon)
        )
    
    # Temporal smoothness loss (if weight > 0)
    smoothness_loss = None
    if smoothness_weight > 0:
        smoothness_loss = TemporalSmoothnessLoss(
            loss_type=smoothness_config.get("type", loss_type),
            epsilon=smoothness_config.get("epsilon", epsilon),
            max_value=smoothness_config.get("max_value", 1e6)
        )
    
    # Joint angle loss (if weight > 0 and configurations provided)
    joint_angle_loss = None
    if angle_weight > 0 and "joint_chains" in angle_config:
        joint_chains = angle_config["joint_chains"]
        angle_limits = angle_config.get("angle_limits")
        joint_angle_loss = JointAngleLoss(
            joint_chains=joint_chains,
            angle_limits=angle_limits,
            loss_type=angle_config.get("type", loss_type),
            epsilon=angle_config.get("epsilon", epsilon)
        )
    
    # Combine into total loss
    return TotalLoss(
        reconstruction_loss=reconstruction_loss,
        consistency_loss=consistency_loss,
        smoothness_loss=smoothness_loss,
        joint_angle_loss=joint_angle_loss,
        recon_weight=recon_weight,
        consistency_weight=consistency_weight,
        smoothness_weight=smoothness_weight,
        angle_weight=angle_weight,
        max_loss=losses_config.get("max_loss", 1e6)
    ) 