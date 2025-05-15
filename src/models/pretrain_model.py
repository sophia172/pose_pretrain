"""
Pretraining model for Human Pose Estimation.

This module defines the main pretraining model architecture for
learning 3D human pose representations from 2D inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .components.encoders import PoseEncoder, PoseDecoder
from .components.attention import PositionalEncoding


class PretrainModel(nn.Module):
    """
    Main pretraining model for Human Pose Estimation.
    
    Implements an encoder-decoder architecture for learning to reconstruct
    3D pose from 2D input. Can be used for unsupervised or supervised pretraining.
    """
    
    def __init__(
        self,
        num_joints: int = 133,
        input_dim: int = 2,  # 2D input
        output_dim: int = 3,  # 3D output
        latent_dim: int = 256,
        encoder_hidden_dims: List[int] = [1024, 512],
        decoder_hidden_dims: List[int] = [512, 1024],
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_joint_relations: bool = True,
        consistency_loss_weight: float = 0.5,
        smoothness_loss_weight: float = 0.1
    ):
        """
        Initialize the pretraining model.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input pose keypoints (2 for 2D poses)
            output_dim: Dimension of output pose keypoints (3 for 3D poses)
            latent_dim: Dimension of latent representations
            encoder_hidden_dims: List of hidden dimensions for encoder
            decoder_hidden_dims: List of hidden dimensions for decoder
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "silu")
            use_positional_encoding: Whether to use positional encoding
            use_joint_relations: Whether to use joint relation attention
            consistency_loss_weight: Weight for consistency loss
            smoothness_loss_weight: Weight for temporal smoothness loss
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Loss weights
        self.consistency_loss_weight = consistency_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight
        
        # Set up encoder (2D to latent)
        self.encoder = PoseEncoder(
            num_joints=num_joints,
            input_dim=input_dim,
            embed_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations
        )
        
        # Get the encoder output dimension
        encoder_output_dim = encoder_hidden_dims[-1] if encoder_hidden_dims else latent_dim
        
        # Set up decoder (latent to 3D)
        self.decoder = PoseDecoder(
            num_joints=num_joints,
            latent_dim=encoder_output_dim,
            output_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            use_joint_relations=use_joint_relations
        )
        
    def forward(
        self, 
        x_in: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x_in:  pose keypoints of shape (batch_size, num_joints, 2) or (batch_size, seq_len, num_joints, 2)
            x_out:  pose keypoints for supervised learning
            return_latent: Whether to return latent representation
            
        Returns:
            Dictionary with model outputs and losses
        """
        # Handle sequential data if needed
        is_sequence = len(x_in.shape) == 4
        batch_size = x_in.shape[0]
        
        if is_sequence:
            # Reshape to (batch_size * seq_len, num_joints, 2)
            seq_len = x_in.shape[1]
            x_in_flat = x_in.reshape(-1, self.num_joints, self.input_dim)
            
            # Pass through encoder
            latent = self.encoder(x_in_flat)
            
            # Pass through decoder
            x_out_pred_flat = self.decoder(latent)
            
            # Reshape back to sequence
            x_out_pred = x_out_pred_flat.reshape(batch_size, seq_len, self.num_joints, self.output_dim)
            
            # Reshape latent if needed
            if return_latent:
                latent = latent.reshape(batch_size, seq_len, self.num_joints, -1)
        else:
            # Standard forward pass for single frames
            latent = self.encoder(x_in)
            x_out_pred = self.decoder(latent)
      
        # Build result dictionary
        result = {
            'pred': x_out_pred,
        }
        
        if return_latent:
            result['latent'] = latent
            
        return result
    
    def _compute_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss based on limb lengths.
        
        Args:
            pred: Predicted 3D poses
            target: Target 3D poses
            
        Returns:
            Consistency loss
        """
        # Define limb connections (pairs of joint indices)
        # This is a simplified example - should be configured for the specific skeleton
        limb_connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
            (8, 13), (13, 14)
        ]
        
        # Calculate limb lengths for predictions and targets
        pred_lengths = []
        target_lengths = []
        
        for joint1, joint2 in limb_connections:
            # Handle different tensor shapes
            if len(pred.shape) == 4:  # batch, seq, joints, dims
                pred_limb = torch.norm(pred[:, :, joint1, :] - pred[:, :, joint2, :], dim=-1)
                target_limb = torch.norm(target[:, :, joint1, :] - target[:, :, joint2, :], dim=-1)
            else:  # batch, joints, dims
                pred_limb = torch.norm(pred[:, joint1, :] - pred[:, joint2, :], dim=-1)
                target_limb = torch.norm(target[:, joint1, :] - target[:, joint2, :], dim=-1)
                
            pred_lengths.append(pred_limb)
            target_lengths.append(target_limb)
            
        # Stack limb lengths
        pred_lengths = torch.stack(pred_lengths, dim=-1)
        target_lengths = torch.stack(target_lengths, dim=-1)
        
        # Compute loss as the difference in limb length ratios
        # This encourages the model to maintain consistent limb proportions
        loss = F.l1_loss(pred_lengths, target_lengths)
        
        return loss
    
    def _compute_smoothness_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal smoothness loss for sequential data.
        
        Args:
            poses: 3D poses of shape (batch_size, seq_len, num_joints, 3)
            
        Returns:
            Smoothness loss
        """
        # Calculate velocity (difference between consecutive frames)
        velocity = poses[:, 1:] - poses[:, :-1]
        
        # Calculate acceleration (difference in velocity)
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # Compute loss as the L1 norm of acceleration
        loss = torch.mean(torch.abs(acceleration))
        
        return loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode pose keypoints to latent representations.
        
        Args:
            x: pose keypoints of shape (batch_size, num_joints, 2) or (batch_size, seq_len, num_joints, 2)
            
        Returns:
            Latent representations
        """
        is_sequence = len(x.shape) == 4
        batch_size = x.shape[0]
        
        if is_sequence:
            # Handle sequential data
            seq_len = x.shape[1]
            x_flat = x.reshape(-1, self.num_joints, self.input_dim)
            latent = self.encoder(x_flat)
            return latent.reshape(batch_size, seq_len, self.num_joints, -1)
        else:
            # Handle single frames
            return self.encoder(x)
            
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to pose keypoints.
        
        Args:
            latent: Latent representations
            
        Returns:
            pose keypoints
        """
        is_sequence = len(latent.shape) == 4
        batch_size = latent.shape[0]
        
        if is_sequence:
            # Handle sequential data
            seq_len = latent.shape[1]
            latent_flat = latent.reshape(-1, self.num_joints, -1)
            x_pred_flat = self.decoder(latent_flat)
            return x_pred_flat.reshape(batch_size, seq_len, self.num_joints, self.output_dim)
        else:
            # Handle single frames
            return self.decoder(latent)
            
    def predict(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Predict pose keypoints from inputs.
        
        Args:
            x_in: 2D or 3D pose keypoints
            
        Returns:
            Predicted 2D or 3D pose keypoints
        """
        return self.forward(x_in)['pred']
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'num_joints': self.num_joints,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'latent_dim': self.latent_dim,
            'consistency_loss_weight': self.consistency_loss_weight,
            'smoothness_loss_weight': self.smoothness_loss_weight
        } 