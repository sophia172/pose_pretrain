"""
Vision Transformer components for keypoint estimation models.

This module contains the Vision Transformer (ViT) components used 
for keypoint-to-keypoint mapping in human pose estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple, Union

from .attention import MultiHeadAttention, PositionalEncoding, JointRelationAttention
from .encoders import TransformerEncoderLayer, TransformerDecoderLayer


class KeypointPatchEmbedding(nn.Module):
    """
    Patch embedding specifically designed for keypoint data.
    
    This module processes keypoint data by treating each joint as a "patch" 
    and projecting it into a higher-dimensional space.
    """
    
    def __init__(
        self,
        num_joints: int,
        input_dim: int,  # 2 for 2D keypoints, 3 for 3D keypoints
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize keypoint patch embedding.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input keypoints (2 for 2D, 3 for 3D)
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Linear projection from input_dim to embed_dim for each joint
        self.projection = nn.Linear(input_dim, embed_dim)
        
        # Class token (CLS) - representation of entire pose
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project keypoints to embedding space and add CLS token.
        
        Args:
            x: Input keypoints of shape (batch_size, num_joints, input_dim)
            
        Returns:
            Patch embeddings with CLS token of shape (batch_size, num_joints+1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Project each joint to embedding dimension
        x = self.projection(x)  # (batch_size, num_joints, embed_dim)
        
        # Expand CLS token to batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with joint embeddings
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_joints+1, embed_dim)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class KeypointViT(nn.Module):
    """
    Vision Transformer model adapted for keypoint-to-keypoint mapping.
    
    This model uses a transformer architecture to map between 2D and 3D keypoints, 
    or to denoise 2D/3D keypoints.
    """
    
    def __init__(
        self,
        num_joints: int,
        input_dim: int,  # 2 for 2D, 3 for 3D
        output_dim: int,  # 2 for 2D, 3 for 3D
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_cls_token: bool = False,
        use_positional_encoding: bool = True,
        use_joint_relations: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize KeypointViT.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input keypoints (2 for 2D, 3 for 3D)
            output_dim: Dimension of output keypoints (2 for 2D, 3 for 3D)
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_cls_token: Whether to use a CLS token and return its representation
            use_positional_encoding: Whether to use positional encoding
            use_joint_relations: Whether to use joint relation attention
            activation: Activation function ("relu", "gelu", "silu")
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = KeypointPatchEmbedding(
            num_joints=num_joints,
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=embed_dim,
                max_len=num_joints + 1,  # +1 for CLS token
                dropout=dropout
            )
        
        # Joint relation attention
        self.use_joint_relations = use_joint_relations
        if use_joint_relations:
            self.joint_relation = JointRelationAttention(
                num_joints=num_joints + 1,  # +1 for CLS token
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attention_dropout
            )
        
        # Get activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=mlp_ratio * embed_dim,
                dropout=dropout,
                activation=self.activation
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head
        if use_cls_token:
            # If using CLS token, output from its representation
            self.head = nn.Linear(embed_dim, num_joints * output_dim)
        else:
            # Otherwise, project each joint representation to output dimension
            self.head = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for KeypointViT.
        
        Args:
            x: Input keypoints of shape (batch_size, num_joints, input_dim)
            
        Returns:
            Output keypoints of shape (batch_size, num_joints, output_dim)
        """
        batch_size = x.shape[0]
        
        # Get patch embeddings with CLS token
        x = self.patch_embed(x)  # (batch_size, num_joints+1, embed_dim)
        
        # Apply positional encoding if used
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # Apply joint relation attention if used
        if self.use_joint_relations:
            x = self.joint_relation(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        if self.use_cls_token:
            # Use CLS token representation
            x = x[:, 0]  # (batch_size, embed_dim)
            x = self.head(x)  # (batch_size, num_joints * output_dim)
            x = x.view(batch_size, self.num_joints, self.output_dim)
        else:
            # Discard CLS token and project each joint
            x = x[:, 1:]  # (batch_size, num_joints, embed_dim)
            x = self.head(x)  # (batch_size, num_joints, output_dim)
        
        return x


class Keypoint2KeypointVIT(nn.Module):
    """
    Vision Transformer model for mapping between keypoint representations.
    
    This model can be configured to map 2D-to-3D, 3D-to-2D, 2D-to-2D, or 3D-to-3D.
    """
    
    def __init__(
        self,
        num_joints: int,
        input_dim: int,  # 2 for 2D, 3 for 3D
        output_dim: int,  # 2 for 2D, 3 for 3D
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_joint_relations: bool = True,
        activation: str = "gelu",
        latent_dim: Optional[int] = None
    ):
        """
        Initialize Keypoint2KeypointVIT.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input keypoints (2 for 2D, 3 for 3D)
            output_dim: Dimension of output keypoints (2 for 2D, 3 for 3D)
            embed_dim: Embedding dimension
            encoder_depth: Number of encoder transformer layers
            decoder_depth: Number of decoder transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_positional_encoding: Whether to use positional encoding
            use_joint_relations: Whether to use joint relation attention
            activation: Activation function ("relu", "gelu", "silu")
            latent_dim: Dimension of latent space (if different from embed_dim)
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Set latent dimension
        if latent_dim is None:
            latent_dim = embed_dim
        
        # Encoder
        self.encoder = KeypointViT(
            num_joints=num_joints,
            input_dim=input_dim,
            output_dim=latent_dim // num_joints,  # Project to latent space
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_cls_token=True,  # Use CLS token for latent representation
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations,
            activation=activation
        )
        
        # Latent projection (if needed)
        if latent_dim != embed_dim:
            self.latent_projection = nn.Linear(latent_dim, embed_dim)
        else:
            self.latent_projection = nn.Identity()
        
        # Decoder
        self.decoder = KeypointViT(
            num_joints=num_joints,
            input_dim=latent_dim // num_joints,  # Input from latent space
            output_dim=output_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_cls_token=False,  # Don't use CLS token in decoder
            use_positional_encoding=use_positional_encoding,
            use_joint_relations=use_joint_relations,
            activation=activation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Keypoint2KeypointVIT.
        
        Args:
            x: Input keypoints of shape (batch_size, num_joints, input_dim)
            
        Returns:
            Output keypoints of shape (batch_size, num_joints, output_dim)
        """
        # Encode input keypoints to latent representation
        latent = self.encoder(x)
        
        # Project latent representation if needed
        latent = self.latent_projection(latent)
        
        # Decode latent representation to output keypoints
        output = self.decoder(latent)
        
        return output 