import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LinearProjector(nn.Module):
    """Simple linear projection from visual features to language model dimensions"""
    def __init__(self, visual_dim: int, language_dim: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(visual_dim, language_dim)
    
    def forward(self, x):
        return self.proj(x)


class MLPProjector(nn.Module):
    """MLP projection with one hidden layer and ReLU activation"""
    def __init__(self, visual_dim: int, language_dim: int, hidden_dim: int = 2048, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """Simple transformer layer with self-attention"""
    def __init__(self, dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention block
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward block
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)
        
        return x


class TransformerProjector(nn.Module):
    """Transformer-based projector with multiple self-attention layers"""
    def __init__(self, visual_dim: int, language_dim: int, num_layers: int = 2,
                 nhead: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        # Project to intermediate dimension for transformer processing
        self.input_proj = nn.Linear(visual_dim, language_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(language_dim, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(language_dim)
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class ConvolutionalProjector(nn.Module):
    """Convolutional projector with multiple kernel sizes for temporal modeling"""
    def __init__(self, visual_dim: int, language_dim: int, kernel_sizes: List[int] = [3, 5],
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        
        # Multi-scale temporal convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(visual_dim, language_dim // len(kernel_sizes), k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.norm = nn.LayerNorm(language_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Convert to (B, C, T) for convolution
        x_conv = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        # Apply temporal convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_output = conv(x_conv)  # (B, C_out, T)
            conv_outputs.append(conv_output)
        
        # Concatenate multi-scale features
        x = torch.cat(conv_outputs, dim=1)  # (B, C_out, T)
        
        # Convert back to (B, T, C)
        x = x.transpose(1, 2)  # (B, C_out, T) -> (B, T, C_out)
        
        # Apply normalization and dropout
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class CrossModalAttentionProjector(nn.Module):
    """Cross-modal attention projector with learnable query vectors"""
    def __init__(self, visual_dim: int, language_dim: int, nhead: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        # Initial projection for visual features
        self.visual_proj = nn.Linear(visual_dim, language_dim)
        
        # Learnable query vectors to represent language model tokens
        self.query = nn.Parameter(torch.randn(1, 1, language_dim))
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(language_dim, nhead, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(language_dim)
        self.norm2 = nn.LayerNorm(language_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(language_dim, language_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim * 4, language_dim)
        )
    
    def forward(self, x):
        # Project visual features
        x = self.visual_proj(x)  # (B, T, C)
        
        # Create query by expanding to batch size
        batch_size, seq_len, _ = x.shape
        query = self.query.expand(1, batch_size, -1)  # (1, B, C)
        
        # Apply cross-attention: query attends to visual features
        visual_key_value = x.transpose(0, 1)  # (T, B, C)
        attn_output, _ = self.cross_attn(query, visual_key_value, visual_key_value)
        
        # Reshape output
        attn_output = attn_output.transpose(0, 1)  # (B, 1, C)
        
        # Apply normalization, residual connection and MLP
        attn_output = self.norm1(attn_output)
        output = attn_output + self.dropout(self.mlp(attn_output))
        output = self.norm2(output)
        
        # Expand to match sequence length of input
        output = output.expand(-1, seq_len, -1)  # (B, T, C)
        
        return output


class GatedProjector(nn.Module):
    """Gated projection mechanism with feature-wise gating"""
    def __init__(self, visual_dim: int, language_dim: int, hidden_dim: int = 2048, 
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        
        # Feature transformation path
        self.feature_path = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim)
        )
        
        # Gate computation path
        self.gate_path = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute features and gates
        features = self.feature_path(x)
        gates = self.gate_path(x)
        
        # Apply gating mechanism
        output = features * gates
        
        return output


def get_projector(projector_type: str, visual_dim: int, language_dim: int, **kwargs):
    """
    Factory function to create the appropriate projector based on configuration.
    
    Args:
        projector_type: Type of projector to use ('linear', 'mlp', 'transformer', etc.)
        visual_dim: Input dimension from visual encoder
        language_dim: Output dimension for language model
        **kwargs: Additional arguments specific to each projector type
    
    Returns:
        A projector module that maps from visual_dim to language_dim
    """
    if projector_type == "linear":
        return LinearProjector(visual_dim, language_dim, **kwargs)
    elif projector_type == "mlp":
        return MLPProjector(visual_dim, language_dim, **kwargs)
    elif projector_type == "transformer":
        return TransformerProjector(visual_dim, language_dim, **kwargs)
    elif projector_type == "convolutional":
        return ConvolutionalProjector(visual_dim, language_dim, **kwargs)
    elif projector_type == "cross_modal_attention":
        return CrossModalAttentionProjector(visual_dim, language_dim, **kwargs)
    elif projector_type == "gated":
        return GatedProjector(visual_dim, language_dim, **kwargs)
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
