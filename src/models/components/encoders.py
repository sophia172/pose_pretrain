"""
Encoder and decoder components for pose estimation models.

This module contains the encoder and decoder components used in the
pretraining model for human pose estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple

from .attention import MultiHeadAttention, PositionalEncoding, JointRelationAttention


class PoseEncoder(nn.Module):
    """
    Encoder for 2D pose keypoints.
    
    Transforms 2D pose keypoints into latent representations that capture
    pose structure and contextual information.
    """
    
    def __init__(
        self,
        num_joints: int,
        input_dim: int = 2,  # 2D input
        embed_dim: int = 256,
        hidden_dims: List[int] = [1024, 512],
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_joint_relations: bool = True
    ):
        """
        Initialize pose encoder.
        
        Args:
            num_joints: Number of joints in the pose
            input_dim: Dimension of input pose keypoints (2 for 2D poses)
            embed_dim: Embedding dimension
            hidden_dims: List of hidden dimensions for feedforward layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "silu")
            use_positional_encoding: Whether to use positional encoding
            use_joint_relations: Whether to use joint relation attention
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection from input_dim to embed_dim for each joint
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=embed_dim,
                max_len=num_joints,
                dropout=dropout
            )
            
        # Joint relation attention
        self.use_joint_relations = use_joint_relations
        if use_joint_relations:
            self.joint_relation = JointRelationAttention(
                num_joints=num_joints,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
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
            
        # Build transformer encoder layers
        encoder_layers = []
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dims[0] if hidden_dims else embed_dim * 4,
                dropout=dropout,
                activation=self.activation
            )
            encoder_layers.append(layer)
            
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Output projection if needed
        if hidden_dims:
            self.output_projection = nn.Linear(embed_dim, hidden_dims[-1])
        else:
            self.output_projection = None
            
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D pose keypoints.
        
        Args:
            x: Input pose keypoints of shape (batch_size, num_joints, input_dim)
            
        Returns:
            Encoded pose features of shape (batch_size, num_joints, embed_dim or hidden_dims[-1])
        """
        batch_size, num_joints, input_dim = x.shape
        
        # Ensure dimensions match expected values
        assert num_joints == self.num_joints, f"Expected {self.num_joints} joints, got {num_joints}"
        assert input_dim == self.input_dim, f"Expected {self.input_dim} input dimensions, got {input_dim}"
        
        # Project each joint to embedding dimension
        x = self.input_projection(x)  # (batch_size, num_joints, embed_dim)
        
        # Apply positional encoding if used
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
            
        # Apply joint relation attention if used
        if self.use_joint_relations:
            x = self.joint_relation(x)
            
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Apply normalization
        x = self.norm(x)
        
        # Apply output projection if needed
        if self.output_projection is not None:
            x = self.output_projection(x)
            
        return x


class PoseDecoder(nn.Module):
    """
    Decoder for transforming latent representations back to 3D pose keypoints.
    
    Transforms latent representations into 3D pose keypoints, incorporating
    structural constraints and contextual information.
    """
    
    def __init__(
        self,
        num_joints: int,
        latent_dim: int,
        output_dim: int = 3,  # 3D output
        hidden_dims: List[int] = [512, 1024],
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_joint_relations: bool = True
    ):
        """
        Initialize pose decoder.
        
        Args:
            num_joints: Number of joints in the pose
            latent_dim: Dimension of latent representations
            output_dim: Dimension of output pose keypoints (3 for 3D poses)
            hidden_dims: List of hidden dimensions for feedforward layers
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "silu")
            use_joint_relations: Whether to use joint relation attention
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Input projection if hidden_dims is provided
        if hidden_dims:
            self.input_projection = nn.Linear(latent_dim, hidden_dims[0])
            working_dim = hidden_dims[0]
        else:
            self.input_projection = None
            working_dim = latent_dim
            
        # Get activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Joint relation attention
        self.use_joint_relations = use_joint_relations
        if use_joint_relations:
            self.joint_relation = JointRelationAttention(
                num_joints=num_joints,
                embed_dim=working_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
        # Build transformer decoder layers
        decoder_layers = []
        for i in range(num_layers):
            # Use the next hidden dimension if available
            
            
            layer = TransformerDecoderLayer(
                embed_dim=working_dim,
                num_heads=num_heads,
                hidden_dim=working_dim * 4,  # Typical FFN size
                dropout=dropout,
                activation=self.activation
            )
            decoder_layers.append(layer)
            
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Output projection to get 3D coordinates
        self.output_projection = nn.Linear(working_dim, output_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(working_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to 3D pose keypoints.
        
        Args:
            x: Latent representations of shape (batch_size, num_joints, latent_dim)
            
        Returns:
            Decoded 3D pose keypoints of shape (batch_size, num_joints, output_dim)
        """
        batch_size, num_joints, latent_dim = x.shape
        
        # Ensure dimensions match expected values
        assert num_joints == self.num_joints, f"Expected {self.num_joints} joints, got {num_joints}"
        assert latent_dim == self.latent_dim, f"Expected {self.latent_dim} latent dimensions, got {latent_dim}"
        
        # Apply input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            
        # Apply joint relation attention if used
        if self.use_joint_relations:
            x = self.joint_relation(x)
            
        # Apply transformer decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
            
        # Apply normalization
        x = self.norm(x)
        
        # Apply output projection to get 3D coordinates
        x = self.output_projection(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention.
    
    Standard transformer encoder layer as described in
    "Attention Is All You Need".
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ):
        """
        Initialize transformer encoder layer.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function module
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            src: Input sequence of shape (batch_size, seq_len, embed_dim)
            src_mask: Mask for self-attention mechanism
            src_key_padding_mask: Mask for padded elements
            
        Returns:
            Output sequence of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention block
        src2, _ = self.self_attn(
            query=self.norm1(src),
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask
        )
        src = src + self.dropout(src2)
        
        # Feedforward block
        src2 = self.feedforward(self.norm2(src))
        src = src + src2
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention and cross-attention.
    
    Standard transformer decoder layer as described in
    "Attention Is All You Need", simplified for pose estimation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ):
        """
        Initialize transformer decoder layer.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function module
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for transformer decoder layer.
        
        Args:
            tgt: Target sequence of shape (batch_size, seq_len, embed_dim)
            memory: Memory sequence from encoder (not used in this simplified version)
            tgt_mask: Mask for self-attention mechanism
            memory_mask: Mask for cross-attention mechanism (not used)
            tgt_key_padding_mask: Mask for padded elements in target
            memory_key_padding_mask: Mask for padded elements in memory (not used)
            
        Returns:
            Output sequence of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention block
        tgt2, _ = self.self_attn(
            query=self.norm1(tgt),
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # Feedforward block
        tgt2 = self.feedforward(self.norm2(tgt))
        tgt = tgt + tgt2
        
        return tgt 