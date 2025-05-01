"""
Vision Transformer-based Pretraining model for Keypoint Estimation.

This module defines the Vision Transformer-based pretraining model architecture
for learning keypoint mapping between different representations (2D-2D, 3D-3D, 2D-3D).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .components.vision_transformer import KeypointViT, Keypoint2KeypointVIT


class PretrainViTModel(nn.Module):
    """
    Vision Transformer-based pretraining model for keypoint mapping.
    
    Implements a Vision Transformer architecture for learning to map between
    different keypoint representations (2D-2D, 3D-3D, or 2D-3D).
    """
    
    def __init__(
        self,
        num_joints: int = 133,
        input_dim: int = 2,  # 2D or 3D input
        output_dim: int = 2,  # 2D or 3D output
        embed_dim: int = 256,
        latent_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_joint_relations: bool = True,
        consistency_loss_weight: float = 0.5,
        smoothness_loss_weight: float = 0.1
    ):
        """
        Initialize the Vision Transformer-based pretraining model.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input pose keypoints (2 for 2D, 3 for 3D)
            output_dim: Dimension of output pose keypoints (2 for 2D, 3 for 3D)
            embed_dim: Embedding dimension for Vision Transformer
            latent_dim: Dimension of latent representations
            encoder_depth: Number of encoder transformer layers
            decoder_depth: Number of decoder transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
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
        
        # Set up the Vision Transformer model for keypoint mapping
        self.model = Keypoint2KeypointVIT(
            num_joints=num_joints,
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations,
            activation=activation,
            latent_dim=latent_dim
        )
        
    def forward(
        self, 
        x_in: torch.Tensor,
        x_target: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x_in: Input pose keypoints of shape (batch_size, num_joints, input_dim) 
                 or (batch_size, seq_len, num_joints, input_dim)
            x_target: Optional target pose keypoints for supervised learning
            return_latent: Whether to return latent representation
            
        Returns:
            Dictionary with model outputs and losses
        """
        # Handle sequential data if needed
        is_sequence = len(x_in.shape) == 4
        batch_size = x_in.shape[0]
        
        if is_sequence:
            # Reshape to (batch_size * seq_len, num_joints, input_dim)
            seq_len = x_in.shape[1]
            x_in_flat = x_in.reshape(-1, self.num_joints, self.input_dim)
            
            # Forward pass through Vision Transformer
            x_out_flat = self.model(x_in_flat)
            
            # Reshape back to sequence
            x_out = x_out_flat.reshape(batch_size, seq_len, self.num_joints, self.output_dim)
        else:
            # Standard forward pass for single frames
            x_out = self.model(x_in)
        
        # Calculate losses if targets provided
        loss_dict = {}
        
        if x_target is not None:
            # Reconstruction loss (L1 loss)
            recon_loss = F.l1_loss(x_out, x_target)
            loss_dict['reconstruction_loss'] = recon_loss
            
            # Add consistency loss if weight > 0
            if self.consistency_loss_weight > 0:
                # Limb length consistency loss
                consistency_loss = self._compute_consistency_loss(x_out, x_target)
                loss_dict['consistency_loss'] = consistency_loss * self.consistency_loss_weight
                
            # Add temporal smoothness loss if weight > 0 and input is a sequence
            if self.smoothness_loss_weight > 0 and is_sequence:
                # Temporal smoothness loss
                smoothness_loss = self._compute_smoothness_loss(x_out)
                loss_dict['smoothness_loss'] = smoothness_loss * self.smoothness_loss_weight
                
            # Compute total loss
            total_loss = recon_loss
            
            if self.consistency_loss_weight > 0:
                total_loss = total_loss + loss_dict['consistency_loss']
                
            if self.smoothness_loss_weight > 0 and is_sequence:
                total_loss = total_loss + loss_dict['smoothness_loss']
                
            loss_dict['total_loss'] = total_loss
        
        # Build result dictionary
        result = {
            'pred': x_out,
            **loss_dict
        }
            
        return result
    
    def _compute_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss based on limb lengths.
        
        Args:
            pred: Predicted poses
            target: Target poses
            
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
            poses: Output poses of shape (batch_size, seq_len, num_joints, output_dim)
            
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
    
    def predict(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Predict output keypoints from input keypoints.
        
        Args:
            x_in: Input pose keypoints of shape (batch_size, num_joints, input_dim)
            
        Returns:
            Output pose keypoints of shape (batch_size, num_joints, output_dim)
        """
        return self.model(x_in)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.
        
        Returns:
            Model configuration dictionary
        """
        return {
            "model_type": "pretrain_vit",
            "num_joints": self.num_joints,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "consistency_loss_weight": self.consistency_loss_weight,
            "smoothness_loss_weight": self.smoothness_loss_weight
        } 