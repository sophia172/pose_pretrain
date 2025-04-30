"""
Attention mechanisms for pose models.

This module contains various attention mechanisms that can be used
in pose estimation models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Self-attention layer implementation.
    
    Implements the self-attention mechanism as described in the
    "Attention Is All You Need" paper.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize self-attention layer.
        
        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to include bias parameters
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot products
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for self-attention layer.
        
        Args:
            query: Query embeddings of shape (batch_size, seq_len, embed_dim)
            key: Key embeddings of shape (batch_size, seq_len, embed_dim)
            value: Value embeddings of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Mask for padded elements in the key
            attn_mask: Mask for attention weights
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Use query as key and value if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim)
        
        # Transpose to shape [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, tgt_len, src_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention masks if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        if key_padding_mask is not None:
            # Expand key_padding_mask to correct shape
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
        
        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        # [batch_size, num_heads, tgt_len, head_dim]
        attn_output = torch.matmul(attn_probs, v)
        
        # Transpose and reshape back to [batch_size, tgt_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        if need_weights:
            # Average attention weights over heads
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention wrapper using PyTorch's implementation.
    
    This class wraps PyTorch's MultiheadAttention module with a more
    convenient interface for use in pose models.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention layer.
        
        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to include bias parameters
        """
        super().__init__()
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True
        )
        
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention layer.
        
        Args:
            query: Query embeddings of shape (batch_size, seq_len, embed_dim)
            key: Key embeddings of shape (batch_size, seq_len, embed_dim)
            value: Value embeddings of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Mask for padded elements in the key
            attn_mask: Mask for attention weights
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Use query as key and value if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Call PyTorch's MultiheadAttention
        return self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights
        )


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds positional information to the input embeddings as described in
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Compute sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class JointRelationAttention(nn.Module):
    """
    Attention mechanism that models joint relationships in human poses.
    
    This module learns relationships between different joints in a human pose
    to better capture structural dependencies.
    """
    
    def __init__(
        self, 
        num_joints: int, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        """
        Initialize joint relation attention.
        
        Args:
            num_joints: Number of joints in the pose
            embed_dim: Dimension of joint embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        
        # Learnable joint embeddings
        self.joint_embeddings = nn.Parameter(torch.randn(num_joints, embed_dim))
        
        # Joint relation attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply joint relation attention to input features.
        
        Args:
            x: Input features of shape (batch_size, num_joints, feature_dim)
            
        Returns:
            Features with joint relations incorporated
        """
        batch_size, num_joints, feature_dim = x.shape
        
        # Ensure number of joints matches expected value
        assert num_joints == self.num_joints, f"Expected {self.num_joints} joints, got {num_joints}"
        
        # Project input features to embedding dimension if needed
        if feature_dim != self.embed_dim:
            x = self.out_proj(x)
            
        # Add learnable joint embeddings
        joint_emb = self.joint_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + joint_emb
        
        # Apply self-attention to capture joint relations
        attn_out, _ = self.attention(x)
        
        # Residual connection, normalization and dropout
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        
        return x 