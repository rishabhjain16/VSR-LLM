# Copyright (c) 2023
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import inspect
import logging

# Get logger
logger = logging.getLogger(__name__)

class BaseProjector(nn.Module):
    """Base class for all projectors"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class LinearProjector(BaseProjector):
    """Simple linear projection layer (baseline)"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        return self.proj(x)  # [B, T, output_dim]


class MLPProjector(BaseProjector):
    """MLP-based projector with configurable depth and activation"""
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        if hidden_dim is None:
            hidden_dim = input_dim * 2
            
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation/dropout after final layer
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                
        self.proj = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        return self.proj(x)  # [B, T, output_dim]


class QFormerProjector(BaseProjector):
    """
    Q-Former style projector, inspired by BLIP-2 architecture
    (https://arxiv.org/pdf/2301.12597.pdf)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        
        # Learnable query tokens with improved initialization
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input projection to handle potential dimension issues
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # Layer norm for better stability
        self.pre_norm = nn.LayerNorm(input_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final projection to match LLM dimensions
        self.proj = nn.Linear(input_dim, output_dim)
        
        # Initialize projection layer properly
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Apply input projection and normalization
        x = self.input_proj(x)
        x = self.pre_norm(x)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Concatenate query tokens with input
        x_with_query = torch.cat([query_tokens, x], dim=1)
        
        # Create attention mask to allow queries to attend to all tokens
        # while input features can only attend to themselves
        query_len, feat_len = self.num_queries, x.size(1)
        total_len = query_len + feat_len
        
        # Create causal attention mask (1 = cannot attend, 0 = can attend)
        # Shape: [total_len, total_len]
        attn_mask = torch.ones(total_len, total_len, device=x.device)
        
        # Allow queries to attend to all tokens
        attn_mask[:query_len, :] = 0
        
        # Allow input features to attend only to themselves
        for i in range(query_len, total_len):
            attn_mask[i, i] = 0
            
        # Convert to proper format for transformer
        attn_mask = attn_mask.bool()
        
        # Get the data type of transformer parameters
        param_dtype = self.transformer.layers[0].norm1.weight.dtype
        
        # Ensure input is in the same data type as transformer parameters
        x_with_query = x_with_query.to(param_dtype)
        
        # Process through transformer
        output = self.transformer(x_with_query, mask=attn_mask)
        
        # Extract query outputs
        query_output = output[:, :query_len]
        
        # Project to target dimension
        projected = self.proj(query_output)
        
        return projected


class CrossAttentionProjector(BaseProjector):
    """
    Cross-attention based projector, inspired by architectures like CLIP and VisualGPT
    (https://arxiv.org/pdf/2304.15010)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        
        # Initialize latent tokens that will attend to visual features
        latent_dim = output_dim
        self.latent_tokens = nn.Parameter(torch.zeros(1, 32, latent_dim))
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        # Input projection to match dimensions if needed
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input if dimensions don't match
        x_proj = self.input_proj(x)
        
        # Expand latent tokens to batch size
        latent = self.latent_tokens.expand(B, -1, -1)
        
        # Apply cross-attention layers
        for layer in self.cross_attn_layers:
            latent = layer(latent, x_proj)
            
        return latent


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for CrossAttentionProjector"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B, Q, D], kv: [B, T, D]
        # Self-attention among queries
        q2 = self.norm1(q)
        q = q + self.cross_attn(q2, kv, kv, need_weights=False)[0]
        
        # FFN
        q = q + self.mlp(self.norm2(q))
        return q


class PerceiverProjector(BaseProjector):
    """
    Perceiver-style projector that compresses variable-length inputs to fixed-length outputs
    (https://arxiv.org/pdf/2103.03206.pdf)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_latents: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        
        # Initialize latent array
        self.latent_dim = output_dim
        self.num_latents = num_latents
        self.latent_array = nn.Parameter(torch.zeros(1, num_latents, self.latent_dim))
        nn.init.normal_(self.latent_array, std=0.02)
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(input_dim, self.latent_dim) if input_dim != self.latent_dim else nn.Identity()
        
        # Cross-attention layers
        self.perceiver_layers = nn.ModuleList([
            PerceiverBlock(self.latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input if needed
        x_proj = self.input_proj(x)
        
        # Expand latent array to batch size
        latent = self.latent_array.expand(B, -1, -1)
        
        # Process through perceiver layers
        for layer in self.perceiver_layers:
            latent = layer(latent, x_proj)
            
        return latent


class PerceiverBlock(nn.Module):
    """Perceiver block for PerceiverProjector"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, latent: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # Cross-attention: latent attends to inputs
        x = self.norm1(latent)
        x = latent + self.cross_attn(x, inputs, inputs, need_weights=False)[0]
        
        # Self-attention: latent attends to itself
        y = self.norm2(x)
        x = x + self.self_attn(y, y, y, need_weights=False)[0]
        
        # FFN
        z = self.norm3(x)
        x = x + self.mlp(z)
        
        return x


class AdaptiveQueryProjector(BaseProjector):
    """
    Adaptive Query Transformer Projector
    Uses learnable query tokens to attend to visual features
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_queries: int = 32
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Input projection with proper initialization
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Add normalization for stability
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable query tokens with improved initialization
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final projection to output dimension with proper initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Store original dtype for consistent return
        input_dtype = x.dtype
        
        # Project input to hidden dimension and normalize
        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = self.input_norm(x)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, hidden_dim]
        
        # Concatenate query tokens with input features
        x_with_queries = torch.cat([query_tokens, x], dim=1)  # [B, num_queries + T, hidden_dim]
        
        # Create attention mask for transformer
        # Shape: [seq_len, seq_len]
        query_len, feat_len = self.num_queries, x.size(1)
        total_len = query_len + feat_len
        
        # Create mask where True (1) means tokens cannot attend to each other
        attn_mask = torch.ones(total_len, total_len, device=x.device)
        
        # Allow queries to attend to all tokens (both queries and inputs)
        attn_mask[:query_len, :] = 0
        
        # Allow input features to attend only to themselves (diagonal)
        for i in range(query_len, total_len):
            attn_mask[i, i] = 0
            
        # Convert to boolean mask for transformer
        attn_mask = attn_mask.bool()
        
        # Get the data type of transformer parameters
        param_dtype = self.transformer.layers[0].norm1.weight.dtype
        
        # Ensure input is in the same data type as transformer parameters
        x_with_queries = x_with_queries.to(param_dtype)
        
        try:
            # Process through transformer
            output = self.transformer(x_with_queries, mask=attn_mask)
            
            # Extract only the query token outputs
            query_output = output[:, :query_len]  # [B, num_queries, hidden_dim]
            
            # Project to output dimension
            result = self.output_proj(query_output)  # [B, num_queries, output_dim]
            
            # Return with original dtype for consistency
            return result.to(input_dtype)
            
        except RuntimeError as e:
            if "expected scalar type" in str(e):
                # Handle mixed precision error explicitly
                logger.warning(f"Mixed precision error in AdaptiveQueryProjector: {e}")
                # Fall back to float32 for the entire computation
                x_with_queries = x_with_queries.to(torch.float32)
                output = self.transformer(x_with_queries, mask=attn_mask)
                query_output = output[:, :query_len]
                result = self.output_proj(query_output)
                return result.to(input_dtype)
            else:
                # Re-raise if it's not a dtype issue
                raise


class FusionRefinementProjector(BaseProjector):
    """
    A novel projector with progressive refinement of features through
    multiple fusion stages - another suggestion for improvement
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_stages: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Progressive refinement stages
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            self.stages.append(
                RefinementStage(output_dim, num_heads, dropout)
            )
            
        # Output tokens that will collect information
        self.output_tokens = nn.Parameter(torch.zeros(1, 32, output_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final integration layer
        self.output_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Progressive refinement
        refined_features = x
        stage_outputs = [refined_features]
        
        for stage in self.stages:
            refined_features = stage(refined_features, stage_outputs)
            stage_outputs.append(refined_features)
            
        # Expand output tokens to batch size
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Final cross-attention to gather information
        normalized = self.norm(output_tokens)
        final_output = output_tokens + self.output_attn(
            normalized, refined_features, refined_features, need_weights=False
        )[0]
        
        return final_output


class RefinementStage(nn.Module):
    """Refinement stage for FusionRefinementProjector"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention to process current features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Integration of previous stage outputs
        self.integration = nn.ModuleDict({
            'attn': nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            'norm': nn.LayerNorm(hidden_dim)
        })
        
        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for selective update
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, previous_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Self-attention processing
        norm_x = self.norm1(x)
        self_attn_out = x + self.self_attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        
        # Integration with previous stages - simple average of previous features
        if len(previous_outputs) > 0:
            prev_context = torch.stack(previous_outputs).mean(dim=0)
            
            # Cross-attention to integrate previous context
            norm_attn = self.integration['norm'](self_attn_out)
            integration_out = self_attn_out + self.integration['attn'](
                norm_attn, prev_context, prev_context, need_weights=False
            )[0]
        else:
            integration_out = self_attn_out
            
        # Feed-forward refinement
        refined = integration_out + self.ffn(self.norm2(integration_out))
        
        # Calculate update gate - how much to keep from original vs. refined
        if len(previous_outputs) > 0:
            gate_input = torch.cat([x, refined], dim=-1)
            gate = self.gate(gate_input)
            output = gate * refined + (1 - gate) * x
        else:
            output = refined
            
        return output


class GatedCrossAttentionProjector(BaseProjector):
    """
    Gated Cross-Attention projector inspired by recent advances in multimodal fusion
    with explicit gating mechanisms to control information flow
    (https://arxiv.org/pdf/2402.10896)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gate_mlp: bool = True
    ):
        super().__init__(input_dim, output_dim)
        
        # Initialize latent tokens
        self.latent_tokens = nn.Parameter(torch.zeros(1, 32, output_dim))
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Cross-attention layers with gating
        self.layers = nn.ModuleList([
            GatedCrossAttentionBlock(
                hidden_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gate_mlp=use_gate_mlp
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input
        x_proj = self.input_proj(x)
        
        # Expand latent tokens
        latent = self.latent_tokens.expand(B, -1, -1)
        
        # Apply layers
        for layer in self.layers:
            latent = layer(latent, x_proj)
            
        return latent


class GatedCrossAttentionBlock(nn.Module):
    """Block with gated cross-attention mechanism"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, use_gate_mlp: bool = True):
        super().__init__()
        self.use_gate_mlp = use_gate_mlp
        
        # Cross-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism
        if use_gate_mlp:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        q_norm = self.norm1(q)
        attn_output = self.cross_attn(q_norm, kv, kv, need_weights=False)[0]
        
        # Compute gating values
        gate_input = torch.cat([q, attn_output], dim=-1)
        gate_values = self.gate(gate_input)
        
        # Apply gate
        gated_output = gate_values * attn_output
        q = q + gated_output
        
        # FFN
        q = q + self.ffn(self.norm2(q))
        return q


class HierarchicalMoEProjector(BaseProjector):
    """
    Hierarchical Mixture-of-Experts projector that combines hierarchical processing
    with sparse expert routing for efficient processing
    (https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f7b1ed763e9c75e4db74b49b76db5f-Paper-Conference.pdf)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dim: int = 2048,
        num_experts: int = 8,
        k: int = 2,  # Top-k experts to route to
        num_layers: int = 2
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hierarchical layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SparseRoutingBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, 32, output_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final attention
        self.output_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Apply MoE layers
        router_logits = []
        for layer in self.layers:
            x, logits = layer(x)
            router_logits.append(logits)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        # Expand output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Apply final attention
        out = output_tokens + self.output_attn(
            self.norm(output_tokens), x, x, need_weights=False
        )[0]
        
        return out


class SparseRoutingBlock(nn.Module):
    """Sparse routing block for MoE Projector"""
    def __init__(self, hidden_dim: int, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        
        # Router
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_dim]
        B, T, D = x.size()
        x_norm = self.norm(x)
        
        # Calculate routing weights
        router_logits = self.router(x_norm)  # [B, T, num_experts]
        
        # Get top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)  # [B, T, k], [B, T, k]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # [B, T, k]
        
        # Initialize output tensor
        expert_outputs = torch.zeros_like(x)  # [B, T, D]
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find positions where this expert is selected (in any of the top-k positions)
            selected_mask = torch.any(top_k_indices == i, dim=-1)  # [B, T]
            if not selected_mask.any():
                continue
                
            # Get batch and token indices where this expert is used
            batch_idx, token_idx = torch.where(selected_mask)
            
            # Get the inputs for this expert
            expert_inputs = x_norm[batch_idx, token_idx]  # [num_selected, D]
            
            # Process inputs through the expert
            expert_output = expert(expert_inputs)  # [num_selected, D]
            
            # For each selected token, find its position in top-k and get corresponding weight
            for idx, (b, t) in enumerate(zip(batch_idx, token_idx)):
                # Find positions in top-k where this expert is selected for this token
                expert_positions = (top_k_indices[b, t] == i).nonzero().squeeze(-1)
                
                # Sum weights for all positions where this expert is selected
                weight = top_k_weights[b, t, expert_positions].sum()
                
                # Apply weighted expert output
                expert_outputs[b, t] += weight * expert_output[idx]
        
        # Residual connection
        output = x + expert_outputs
        
        return output, router_logits


class EnhancedQFormerProjector(BaseProjector):
    """
    Enhanced Q-Former projector inspired by MMS-LLama implementation with
    bidirectional attention and deeper transformer
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,  # Deeper transformer
        num_heads: int = 8,
        intermediate_dim: int = 3072,  # Larger FFN
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Transformer encoder layers with larger FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final projection
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Concatenate query tokens with input features
        x_with_query = torch.cat([query_tokens, x], dim=1)
        
        # Process through transformer
        output = self.transformer(x_with_query)
        
        # Extract only query outputs
        query_output = output[:, :self.num_queries]
        
        # Project to target dimension
        projected = self.proj(query_output)
        
        return projected


class MultiScaleContrastiveProjector(BaseProjector):
    """
    A novel projector that uses multi-scale feature extraction with 
    contrastive alignment between visual and language spaces
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 2048,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        contrastive_temp: float = 0.07
    ):
        super().__init__(input_dim, output_dim)
        self.contrastive_temp = contrastive_temp
        
        # Multi-scale feature extraction
        self.scales = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 2**i
            if kernel_size > 1:
                self.scales.append(nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=kernel_size),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                ))
            else:
                self.scales.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                ))
        
        # Feature fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Contrastive alignment head
        self.align_visual = nn.Linear(hidden_dim, hidden_dim)
        self.align_text = nn.Linear(output_dim, hidden_dim)
        
        # Learnable output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, 32, hidden_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Multi-scale feature extraction
        scale_features = []
        for i, scale in enumerate(self.scales):
            if i == 0:  # First scale is just linear projection
                scale_feat = scale(x)
            else:  # Other scales use 1D convolution
                x_permuted = x.permute(0, 2, 1)  # [B, input_dim, T]
                scale_feat = scale(x_permuted).permute(0, 2, 1)  # [B, T', hidden_dim]
            scale_features.append(scale_feat)
        
        # Pad all scales to the same length
        max_len = max(feat.size(1) for feat in scale_features)
        padded_features = []
        for feat in scale_features:
            if feat.size(1) < max_len:
                padding = feat.new_zeros(B, max_len - feat.size(1), feat.size(2))
                padded_features.append(torch.cat([feat, padding], dim=1))
            else:
                padded_features.append(feat)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(padded_features, dim=1)
        
        # Expand output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Apply fusion attention
        fused_features = output_tokens + self.fusion(
            self.fusion_norm(output_tokens),
            multi_scale_features,
            multi_scale_features,
            need_weights=False
        )[0]
        
        # Project to output dimension
        output = self.output_proj(fused_features)
        
        return output
        
    def compute_contrastive_loss(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Computes contrastive loss between visual and text features for alignment"""
        # Project features to alignment space
        visual_emb = self.align_visual(visual_features)  # [B, N, D]
        text_emb = self.align_text(text_features)        # [B, M, D]
        
        # Normalize embeddings
        visual_emb = F.normalize(visual_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.bmm(visual_emb, text_emb.transpose(1, 2)) / self.contrastive_temp
        
        # Contrastive loss
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)
        
        return loss / 2.0


class PEVLAdapter(BaseProjector):
    """
    Parameter-Efficient Visual-Language Adapter inspired by recent parameter-efficient
    adapters for multimodal tasks
    (https://arxiv.org/pdf/2308.10159.pdf)
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        bottleneck_dim: int = 256,
        num_layers: int = 2,
        num_tasks: int = 1  # For multi-task learning
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Task-specific prompts
        self.task_prompts = nn.Parameter(torch.zeros(num_tasks, 1, 16, output_dim))
        nn.init.normal_(self.task_prompts, std=0.02)
        
        # Low-rank adapters
        self.adapters = nn.ModuleList()
        for _ in range(num_layers):
            self.adapters.append(
                LowRankAdapter(output_dim, bottleneck_dim)
            )
        
        # Output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, 32, output_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final attention
        self.output_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Get task-specific prompt
        prompt = self.task_prompts[task_id].expand(B, -1, -1)
        
        # Concatenate prompt with input
        x_with_prompt = torch.cat([prompt, x], dim=1)
        
        # Apply adapters
        for adapter in self.adapters:
            x_with_prompt = adapter(x_with_prompt)
        
        # Expand output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Apply final attention
        final_output = output_tokens + self.output_attn(
            self.norm(output_tokens),
            x_with_prompt,
            x_with_prompt,
            need_weights=False
        )[0]
        
        return final_output


class LowRankAdapter(nn.Module):
    """Low-rank adapter module for PEVL"""
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x + residual


class SelfAggregatingLinearProjector(BaseProjector):
    """
    Linear projector that includes its own aggregation mechanism,
    producing a fixed number of output tokens like query-based projectors.
    This allows it to be used without external cluster-based aggregation.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        self.num_output_tokens = num_output_tokens
        
        # Linear projection 
        self.proj = nn.Linear(input_dim, output_dim)
        
        # Self-attention for aggregation
        self.query_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, output_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input to output dimension
        x = self.proj(x)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Use cross-attention to aggregate features
        attn_output = query_tokens + self.cross_attention(
            query=self.norm(query_tokens),
            key=x,
            value=x,
            need_weights=False
        )[0]
        
        return attn_output


class SelfAggregatingMLPProjector(BaseProjector):
    """
    MLP projector that includes its own aggregation mechanism,
    producing a fixed number of output tokens like query-based projectors.
    This allows it to be used without external cluster-based aggregation.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        self.num_output_tokens = num_output_tokens
        
        if hidden_dim is None:
            hidden_dim = input_dim * 2
            
        # MLP layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation/dropout after final layer
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                
        self.proj = nn.Sequential(*layers)
        
        # Self-attention for aggregation
        self.query_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, output_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input using MLP
        x = self.proj(x)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Use cross-attention to aggregate features
        attn_output = query_tokens + self.cross_attention(
            query=self.norm(query_tokens),
            key=x,
            value=x,
            need_weights=False
        )[0]
        
        return attn_output


class BLIP2QFormer(BaseProjector):
    """
    Proper implementation of the QFormer architecture from BLIP-2 paper
    (https://arxiv.org/pdf/2301.12597.pdf) with cross-modal attention
    between text instructions and visual/audio features.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        intermediate_size: int = 3072,
        use_bert_config: bool = True
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        
        # Import required modules for BERT-based QFormer
        from transformers import BertConfig, BertLMHeadModel, BertModel
        from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertAttention
        
        # Learnable query tokens with better initialization strategy
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Visual/audio input projection with explicit initialization
        self.input_proj = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Create BERT-based QFormer
        if use_bert_config:
            # Load default BERT configuration 
            qformer_config = BertConfig.from_pretrained("bert-base-uncased")
            qformer_config.encoder_width = input_dim
            qformer_config.num_hidden_layers = num_layers
            qformer_config.num_attention_heads = num_heads
            qformer_config.hidden_size = input_dim
            qformer_config.intermediate_size = intermediate_size
            qformer_config.hidden_dropout_prob = dropout
            qformer_config.attention_probs_dropout_prob = dropout
            
            # Important: Don't set add_cross_attention here
            # We'll handle cross-attention manually
            qformer_config.add_cross_attention = False
            qformer_config.is_decoder = False  
            
            # Initialize the BERT model for self-attention
            self.bert_model = BertModel(qformer_config, add_pooling_layer=False)
            
            # Create separate cross-attention layers with proper initialization
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers // 2)  # Apply cross-attention every 2 layers
            ])
            
            self.cross_layer_norms = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in range(num_layers // 2)
            ])
            
            # Text encoder for instruction tokens (optional)
            # We'll use the same BERT model's embedding layer
            self.use_text_encoder = True
            
        else:
            # Custom implementation using PyTorch modules (alternative)
            self.qformer_layers = nn.ModuleList([
                QFormerLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=input_dim * 4,
                    dropout=dropout,
                    cross_attention=(i % 2 == 0)  # Cross-attn every 2 layers
                ) 
                for i in range(num_layers)
            ])
        
        # Final projection to match LLM dimensions with proper initialization
        self.output_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        # Flag to track if we're using BERT or custom implementation
        self.use_bert_config = use_bert_config
    
    def forward(self, x, text_tokens=None, text_mask=None):
        """
        Forward pass for BLIP-2 style QFormer
        
        Args:
            x: Visual/audio features [B, T, input_dim]
            text_tokens: Optional text token ids [B, L]
            text_mask: Optional attention mask for text [B, L]
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        # Safety check for text inputs
        if text_tokens is not None and not isinstance(text_tokens, torch.Tensor):
            logger.warning(f"text_tokens is not a tensor, but {type(text_tokens)}. Ignoring text input.")
            text_tokens = None
            text_mask = None
        
        # Create mask if needed and text tokens are provided
        if text_tokens is not None and text_mask is None:
            text_mask = (text_tokens != 0).float()  # Convert to float for attention mask
        
        # Get the dtype of the input for consistent handling
        input_dtype = x.dtype
        
        if self.use_bert_config:
            # Run forward pass with consistent dtype handling
            output = self.forward_bert(x, text_tokens, text_mask)
            # Return with original input dtype for consistency
            return output.to(input_dtype)
        else:
            output = self.forward_custom(x, text_tokens, text_mask)
            return output.to(input_dtype)
    
    def forward_bert(self, visual_features, text_input_ids=None, text_attention_mask=None):
        """BERT-based implementation forward pass"""
        B = visual_features.size(0)
        
        # Apply input normalization and projection
        visual_features = self.input_norm(visual_features)
        visual_features = self.input_proj(visual_features)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Get text embeddings if provided
        text_embeds = None
        has_text_input = text_input_ids is not None and self.use_text_encoder
        
        if has_text_input:
            # Use BERT embeddings for text
            with torch.no_grad():  # We don't need to update the embedding layer
                # Handle potential dtype mismatch
                text_input_ids = text_input_ids.to(dtype=torch.long)
                # Get embeddings
                text_embeds = self.bert_model.embeddings.word_embeddings(text_input_ids)
                # Match dtype with query tokens
                text_embeds = text_embeds.to(query_tokens.dtype)
        
        # Ensure consistent dtype throughout the forward pass
        # Get BERT's parameter dtype
        bert_dtype = next(self.bert_model.parameters()).dtype
        query_tokens = query_tokens.to(bert_dtype)
        
        # Process query tokens through BERT layers with custom cross-attention
        hidden_states = query_tokens
        cross_attn_idx = 0
        
        # Apply self-attention from BERT encoder with interleaved cross-attention
        for i, layer in enumerate(self.bert_model.encoder.layer):
            # Apply BERT self-attention layer
            hidden_states = layer(hidden_states)[0]
            
            # Apply cross-attention every 2 layers
            if i % 2 == 1 and cross_attn_idx < len(self.cross_attention_layers):
                # Cross-attention to visual features
                norm_hidden = self.cross_layer_norms[cross_attn_idx](hidden_states)
                
                # Ensure visual features match dtype
                visual_features_matched = visual_features.to(norm_hidden.dtype)
                
                visual_attn_output = self.cross_attention_layers[cross_attn_idx](
                    query=norm_hidden,
                    key=visual_features_matched,
                    value=visual_features_matched,
                    need_weights=False
                )[0]
                hidden_states = hidden_states + visual_attn_output
                
                # Cross-attention to text if available
                if has_text_input:
                    norm_hidden = self.cross_layer_norms[cross_attn_idx](hidden_states)
                    
                    # Ensure text embeddings match dtype 
                    text_embeds_matched = text_embeds.to(norm_hidden.dtype)
                    text_mask_expanded = None
                    
                    if text_attention_mask is not None:
                        text_mask_expanded = ~text_attention_mask.bool()
                    
                    text_attn_output = self.cross_attention_layers[cross_attn_idx](
                        query=norm_hidden,
                        key=text_embeds_matched,
                        value=text_embeds_matched,
                        key_padding_mask=text_mask_expanded,
                        need_weights=False
                    )[0]
                    hidden_states = hidden_states + text_attn_output
                
                cross_attn_idx += 1
        
        # Get query outputs from last hidden states
        query_output = hidden_states
        
        # Project to output dimension
        output = self.output_proj(query_output)
        
        return output
    
    def forward_custom(self, visual_features, text_embeds=None, attention_mask=None):
        """Custom implementation forward pass if not using BERT-based QFormer"""
        B = visual_features.size(0)
        
        # Apply input normalization and projection
        visual_features = self.input_norm(visual_features)
        visual_features = self.input_proj(visual_features)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Process through QFormer layers
        for layer in self.qformer_layers:
            # Only pass text if available
            if text_embeds is not None:
                # Ensure dtype consistency
                text_embeds_matched = text_embeds.to(query_tokens.dtype)
                query_tokens = layer(
                    query_tokens, 
                    visual_features,
                    text_embeds_matched,
                    text_attention_mask=attention_mask
                )
            else:
                query_tokens = layer(query_tokens, visual_features)
        
        # Project to output dimension
        output = self.output_proj(query_tokens)
        
        return output


class QFormerLayer(nn.Module):
    """Custom QFormer layer with self-attention and optional cross-attention"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, cross_attention=True):
        super().__init__()
        # Self-attention for queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention to visual features
        self.has_cross_attn = cross_attention
        if cross_attention:
            self.cross_attn_visual = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.norm_visual = nn.LayerNorm(d_model)
            self.dropout_visual = nn.Dropout(dropout)
            
            # Cross-attention to text (if available)
            self.cross_attn_text = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.norm_text = nn.LayerNorm(d_model)
            self.dropout_text = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, query, visual, text=None, visual_attention_mask=None, text_attention_mask=None):
        # Self-attention
        q = self.norm1(query)
        queries = query + self.dropout1(self.self_attn(q, q, q, need_weights=False)[0])
        
        # Cross-attention to visual features
        if self.has_cross_attn and visual is not None:
            q = self.norm_visual(queries)
            queries = queries + self.dropout_visual(
                self.cross_attn_visual(
                    q, visual, visual, 
                    key_padding_mask=visual_attention_mask,
                    need_weights=False
                )[0]
            )
            
            # Cross-attention to text (if provided)
            if self.has_cross_attn and text is not None and text_attention_mask is not None:
                q = self.norm_text(queries)
                queries = queries + self.dropout_text(
                    self.cross_attn_text(
                        q, text, text,
                        key_padding_mask=text_attention_mask,
                        need_weights=False
                    )[0]
                )
        
        # Feed-forward network
        queries = queries + self.dropout(self.ffn(self.norm2(queries)))
        
        return queries


class TextAwareCrossAttentionProjector(BaseProjector):
    """
    Text-aware version of CrossAttentionProjector that incorporates
    instruction text to guide the cross-attention mechanism
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        
        # Project visual features
        self.visual_proj = nn.Linear(input_dim, output_dim)
        
        # Learnable output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, output_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match output dimension
        self.text_proj = nn.Linear(768, output_dim)  # BERT hidden size is 768
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            TextAwareCrossAttentionLayer(
                d_model=output_dim,
                nhead=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project visual features
        visual_feats = self.visual_proj(x)
        
        # Process text if available
        text_feats = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to output dimension
            text_feats = self.text_proj(text_embeds)
        
        # Expand latent tokens to batch size
        latent_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Process through cross-attention layers
        for layer in self.layers:
            latent_tokens = layer(
                latent_tokens, 
                visual_feats, 
                text_feats,
                visual_attention_mask=None,
                text_attention_mask=text_mask
            )
        
        # Apply final normalization
        output_tokens = self.norm(latent_tokens)
        
        return output_tokens


class TextAwareCrossAttentionLayer(nn.Module):
    """Cross-attention layer that processes visual and text inputs"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Self-attention for output tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention to text
        self.text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, visual, text=None, visual_attention_mask=None, text_attention_mask=None):
        # Self-attention
        q = self.norm1(x)
        x = x + self.dropout(self.self_attn(q, q, q, need_weights=False)[0])
        
        # Cross-attention to visual features
        q = self.norm2(x)
        x = x + self.dropout(
            self.visual_attn(
                q, visual, visual,
                key_padding_mask=visual_attention_mask,
                need_weights=False
            )[0]
        )
        
        # Cross-attention to text if available
        if text is not None:
            q = self.norm3(x)
            x = x + self.dropout(
                self.text_attn(
                    q, text, text,
                    key_padding_mask=None if text_attention_mask is None else ~text_attention_mask.bool(),
                    need_weights=False
                )[0]
            )
        
        # Feed-forward network
        x = x + self.dropout(self.ffn(self.norm4(x)))
        
        return x


class TextAwarePerceiverProjector(BaseProjector):
    """
    Text-aware version of PerceiverProjector that processes both
    visual features and instruction text
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_latents: int = 32,
        num_latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        self.num_latents = num_latents
        
        # Learnable latent vectors
        self.latent_tokens = nn.Parameter(torch.zeros(1, num_latents, num_latent_dim))
        nn.init.normal_(self.latent_tokens, std=0.02)
        
        # Project visual features to latent dimension
        self.visual_proj = nn.Linear(input_dim, num_latent_dim)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match latent dimension
        self.text_proj = nn.Linear(768, num_latent_dim)  # BERT hidden size is 768
        
        # Self-attention blocks for processing latents
        self.layers = nn.ModuleList([
            TextAwarePerceiverLayer(
                d_model=num_latent_dim,
                nhead=num_heads, 
                dim_feedforward=num_latent_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(num_latent_dim, output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project visual features to latent dimension
        visual_feats = self.visual_proj(x)
        
        # Process text if available
        text_feats = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to latent dimension
            text_feats = self.text_proj(text_embeds)
        
        # Expand latent tokens to batch size
        latent_tokens = self.latent_tokens.expand(B, -1, -1)
        
        # Process through perceiver layers
        for layer in self.layers:
            latent_tokens = layer(
                latent_tokens, 
                visual_feats, 
                text_feats,
                visual_attention_mask=None,
                text_attention_mask=text_mask
            )
        
        # Project to output dimension
        output = self.output_proj(latent_tokens)
        
        return output


class TextAwarePerceiverLayer(nn.Module):
    """Perceiver layer that processes latent tokens with cross-attention to inputs"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Self-attention for latent tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention to text
        self.text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latents, visual, text=None, visual_attention_mask=None, text_attention_mask=None):
        # Self-attention
        q = self.norm1(latents)
        latents = latents + self.dropout(self.self_attn(q, q, q, need_weights=False)[0])
        
        # Cross-attention to visual features
        q = self.norm2(latents)
        latents = latents + self.dropout(
            self.visual_attn(
                query=q,
                key=visual,
                value=visual,
                key_padding_mask=visual_attention_mask,
                need_weights=False
            )[0]
        )
        
        # Cross-attention to text if available
        if text is not None:
            q = self.norm3(latents)
            latents = latents + self.dropout(
                self.text_attn(
                    query=q,
                    key=text,
                    value=text,
                    key_padding_mask=None if text_attention_mask is None else ~text_attention_mask.bool(),
                    need_weights=False
                )[0]
            )
        
        # Feed-forward network
        latents = latents + self.dropout(self.ffn(self.norm4(latents)))
        
        return latents


class TextAwareAdaptiveQueryProjector(BaseProjector):
    """
    Text-aware version of the AdaptiveQueryProjector that conditions queries based on text
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_queries: int = 32
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Input projection with proper initialization
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable query tokens with better initialization
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Text encoder - project text tokens to hidden_dim with proper initialization
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
        
        # Text normalization
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention for text conditioning
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final projection with proper initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Store input dtype for consistent return
        input_dtype = x.dtype
        
        # Project input features to hidden_dim and normalize
        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = self.input_norm(x)
        
        # Expand query tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, hidden_dim]
        
        # Process text if available
        if text_tokens is not None:
            # Project text to hidden dimension and normalize
            text_features = self.text_proj(text_tokens)  # [B, L, hidden_dim]
            text_features = self.text_norm(text_features)
            
            # Ensure text features have the same dtype as queries
            text_features = text_features.to(queries.dtype)
            
            # Condition queries based on text using cross-attention
            queries_norm = self.text_cross_norm(queries)
            
            # Create text attention mask if provided
            text_attn_mask = None
            if text_mask is not None:
                # Convert mask to appropriate format for PyTorch attention
                # (True where tokens should be ignored)
                text_attn_mask = ~text_mask.bool()
                
            # Apply cross-attention between queries and text
            queries = queries + self.text_cross_attn(
                query=queries_norm,
                key=text_features,
                value=text_features,
                key_padding_mask=text_attn_mask,
                need_weights=False
            )[0]
        
        # Concatenate query tokens with input features
        x_with_queries = torch.cat([queries, x], dim=1)  # [B, num_queries + T, hidden_dim]
        
        # Get the data type of transformer parameters
        param_dtype = self.transformer.layers[0].norm1.weight.dtype
        
        # Ensure input is in the same data type as transformer parameters
        x_with_queries = x_with_queries.to(param_dtype)
        
        # Create attention mask for the transformer
        # Shape: [seq_len, seq_len] where seq_len = num_queries + input_length
        query_len, feat_len = self.num_queries, x.size(1)
        total_len = query_len + feat_len
        
        # Create attention mask where 1 = cannot attend, 0 = can attend
        # We create a causal mask for the fixed size attention
        attn_mask = torch.ones(total_len, total_len, device=x.device)
        
        # Allow queries to attend to all tokens (both queries and inputs)
        attn_mask[:query_len, :] = 0
        
        # Allow input features to attend only to themselves
        for i in range(query_len, total_len):
            attn_mask[i, i] = 0
            
        # Convert to proper format for transformer
        attn_mask = attn_mask.bool()
        
        # Process through transformer with proper mask
        output = self.transformer(x_with_queries, mask=attn_mask)
        
        # Extract query outputs only
        query_output = output[:, :query_len]  # [B, num_queries, hidden_dim]
        
        # Project to output dimension
        output = self.output_proj(query_output)  # [B, num_queries, output_dim]
        
        # Return with consistent dtype
        return output.to(input_dtype)


class TextAwareHierarchicalMoEProjector(BaseProjector):
    """
    Text-aware version of the Hierarchical Mixture-of-Experts projector
    that uses instruction text to guide expert routing
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dim: int = 2048,
        num_experts: int = 8,
        k: int = 2,  # Top-k experts to route to
        num_layers: int = 2,
        num_output_tokens: int = 32,
        dropout: float = 0.1,
        num_heads: int = 8
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match hidden dimension
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT hidden size is 768
        
        # Text-conditioned router generator
        self.text_router_generator = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Hierarchical MoE layers that consider text
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TextAwareSparseRoutingBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                dropout=dropout
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Output tokens for final cross-attention
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, output_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final attention to aggregate features
        self.output_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Process text if available
        text_features = None
        text_router_bias = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to hidden dimension
            text_features = self.text_proj(text_embeds)
            
            # Create text-based router bias
            # Average text embeddings to get a single vector per batch
            text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(text_embeds)
            text_avg = (text_embeds * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)  # [B, 768]
            
            # Generate router bias from text
            text_router_bias = self.text_router_generator(text_avg)  # [B, hidden_dim]
        
        # Apply MoE layers with text conditioning
        router_logits = []
        for layer in self.layers:
            x, logits = layer(x, text_features, text_router_bias, text_mask)
            router_logits.append(logits)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        # Expand output tokens and apply final attention
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Apply final attention to aggregate features
        out = output_tokens + self.output_attn(
            self.norm(output_tokens), x, x, need_weights=False
        )[0]
        
        return out


class TextAwareSparseRoutingBlock(nn.Module):
    """Text-aware sparse routing block for MoE Projector"""
    def __init__(self, hidden_dim: int, num_experts: int = 8, k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        
        # Router
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Cross-attention to text
        self.text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.text_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, text_features=None, text_router_bias=None, text_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_dim]
        B, T, D = x.size()
        
        # Apply cross-attention to text if available
        if text_features is not None:
            x_norm = self.text_attn_norm(x)
            x = x + self.text_attn(
                query=x_norm,
                key=text_features,
                value=text_features,
                key_padding_mask=None if text_mask is None else ~text_mask.bool(),
                need_weights=False
            )[0]
        
        # Normalize for routing
        x_norm = self.norm(x)
        
        # Calculate routing weights, using text bias if available
        router_logits = self.router(x_norm)  # [B, T, num_experts]
        
        # Add text-based router bias if provided
        if text_router_bias is not None:
            # text_router_bias: [B, hidden_dim]
            # Project to router logits dimensions
            text_bias = text_router_bias @ self.router.weight.t()  # [B, num_experts]
            
            # Add as a bias term to router logits
            router_logits = router_logits + text_bias.unsqueeze(1)
        
        # Get top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)  # [B, T, num_experts]
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)  # [B, T, k], [B, T, k]
        
        # Normalize weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # [B, T, k]
        
        # Initialize output tensor
        expert_outputs = torch.zeros_like(x)  # [B, T, D]
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find positions where this expert is selected (in any of the top-k positions)
            selected_mask = torch.any(top_k_indices == i, dim=-1)  # [B, T]
            if not selected_mask.any():
                continue
                
            # Get batch and token indices where this expert is used
            batch_idx, token_idx = torch.where(selected_mask)
            
            # Get the inputs for this expert
            expert_inputs = x_norm[batch_idx, token_idx]  # [num_selected, D]
            
            # Process inputs through the expert
            expert_output = expert(expert_inputs)  # [num_selected, D]
            
            # For each selected token, find its position in top-k and get corresponding weight
            for idx, (b, t) in enumerate(zip(batch_idx, token_idx)):
                # Find positions in top-k where this expert is selected for this token
                expert_positions = (top_k_indices[b, t] == i).nonzero().squeeze(-1)
                
                # Sum weights for all positions where this expert is selected
                weight = top_k_weights[b, t, expert_positions].sum()
                
                # Apply weighted expert output
                expert_outputs[b, t] += weight * expert_output[idx]
        
        # Residual connection
        output = x + expert_outputs
        
        return output, router_logits


class TextAwareFusionRefinementProjector(BaseProjector):
    """
    Text-aware version of FusionRefinementProjector that uses
    instruction text to guide the refinement process
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_refinement_steps: int = 3,
        dropout: float = 0.1,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match hidden dimension
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT hidden size is 768
        
        # Refinement steps
        self.refinement_steps = nn.ModuleList([
            TextAwareRefinementBlock(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout
            )
            for _ in range(num_refinement_steps)
        ])
        
        # Output tokens for final representation
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, hidden_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Process text if available
        text_features = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to hidden dimension
            text_features = self.text_proj(text_embeds)
        
        # Apply refinement steps
        for refinement_step in self.refinement_steps:
            x = refinement_step(x, text_features, text_mask)
        
        # Generate output representation using output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Cross-attention from output tokens to refined features
        for refinement_step in self.refinement_steps:
            output_tokens = refinement_step(output_tokens, x, None, is_output_tokens=True)
            
            # Also attend to text if available
            if text_features is not None:
                output_tokens = refinement_step(output_tokens, text_features, text_mask, is_output_tokens=True)
        
        # Final projection
        output = self.output_proj(output_tokens)
        
        return output


class TextAwareRefinementBlock(nn.Module):
    """Refinement block with cross-attention to both visual and text features"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None, context_mask=None, is_output_tokens=False):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        
        # Cross-attention if context is provided
        if context is not None:
            residual = x
            x = self.norm2(x)
            x = residual + self.dropout(
                self.cross_attn(
                    query=x,
                    key=context,
                    value=context,
                    key_padding_mask=None if context_mask is None else ~context_mask.bool(),
                    need_weights=False
                )[0]
            )
        
        # Feed-forward network
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x


class TextAwareGatedCrossAttentionProjector(BaseProjector):
    """
    Text-aware version of GatedCrossAttentionProjector where the gating
    mechanism is conditioned on the instruction text
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 3,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match hidden dimension
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT hidden size is 768
        
        # Text-conditioned gating network
        self.text_gate_generator = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gated cross-attention layers
        self.layers = nn.ModuleList([
            TextAwareGatedCrossAttentionLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, hidden_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Process text if available
        text_features = None
        text_gate_bias = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project text for cross-attention
            text_features = self.text_proj(text_embeds)
            
            # Create text-based gating bias
            # Average text embeddings to get a single vector per batch
            text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(text_embeds)
            text_avg = (text_embeds * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)  # [B, 768]
            
            # Generate gate bias from text
            text_gate_bias = self.text_gate_generator(text_avg)  # [B, hidden_dim]
        
        # Initialize output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Process through gated cross-attention layers
        for layer in self.layers:
            output_tokens = layer(
                output_tokens, 
                x, 
                text_features,
                text_gate_bias,
                visual_attention_mask=None,
                text_attention_mask=text_mask
            )
        
        # Final projection to output dimension
        output = self.output_proj(output_tokens)
        
        return output


class TextAwareGatedCrossAttentionLayer(nn.Module):
    """Gated cross-attention layer with text conditioning"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to visual features
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention to text features
        self.text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gating networks
        self.visual_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, visual, text=None, text_gate_bias=None, visual_attention_mask=None, text_attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        
        # Cross-attention to visual features with gating
        if visual is not None:
            residual = x
            x_norm = self.norm2(x)
            
            # Apply cross-attention
            visual_attn_output = self.visual_attn(
                query=x_norm,
                key=visual,
                value=visual,
                key_padding_mask=visual_attention_mask,
                need_weights=False
            )[0]
            
            # Compute gate values
            gate = self.visual_gate(x_norm)
            
            # Apply text-based gating bias if provided
            if text_gate_bias is not None:
                # Expand text gate bias to match x dimensions
                expanded_bias = text_gate_bias.unsqueeze(1).expand(-1, x.size(1), -1)
                gate = gate + expanded_bias.sigmoid() * 0.1  # Add as a small bias
                gate = gate.clamp(0, 1)  # Clamp to [0, 1]
            
            # Apply gated residual connection
            x = residual + self.dropout(gate * visual_attn_output)
        
        # Cross-attention to text features with gating
        if text is not None:
            residual = x
            x_norm = self.norm3(x)
            
            # Apply cross-attention
            text_attn_output = self.text_attn(
                query=x_norm,
                key=text,
                value=text,
                key_padding_mask=None if text_attention_mask is None else ~text_attention_mask.bool(),
                need_weights=False
            )[0]
            
            # Compute gate values
            gate = self.text_gate(x_norm)
            
            # Apply gated residual connection
            x = residual + self.dropout(gate * text_attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm4(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x


class TextAwareMultiScaleContrastiveProjector(BaseProjector):
    """
    Text-aware version of MultiScaleContrastiveProjector that uses
    instruction text as an additional modality in the contrastive objective
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 2048,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        contrastive_temp: float = 0.07,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        self.contrastive_temp = contrastive_temp
        self.num_output_tokens = num_output_tokens
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match hidden dimension
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT hidden size is 768
        
        # Multi-scale feature extraction for visual features
        self.visual_scales = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 2**i
            if kernel_size > 1:
                self.visual_scales.append(nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=kernel_size),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                ))
            else:
                self.visual_scales.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                ))
        
        # Text-conditioned feature fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Contrastive alignment heads
        self.align_visual = nn.Linear(hidden_dim, hidden_dim)
        self.align_text = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, hidden_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Cross-modal fusion transformer
        self.cross_modal_layers = nn.ModuleList([
            TextAwareMultiScaleLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout
            )
            for _ in range(3)  # Use 3 layers for cross-modal fusion
        ])
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Multi-scale feature extraction for visual input
        scale_features = []
        for i, scale in enumerate(self.visual_scales):
            if i == 0:  # First scale is just linear projection
                scale_feat = scale(x)
            else:  # Other scales use 1D convolution
                x_permuted = x.permute(0, 2, 1)  # [B, input_dim, T]
                scale_feat = scale(x_permuted).permute(0, 2, 1)  # [B, T', hidden_dim]
            scale_features.append(scale_feat)
        
        # Process text if available
        text_features = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to hidden dimension
            text_features = self.text_proj(text_embeds)  # [B, L, hidden_dim]
        
        # Pad all scales to the same length
        max_len = max(feat.size(1) for feat in scale_features)
        padded_features = []
        for feat in scale_features:
            if feat.size(1) < max_len:
                padding = feat.new_zeros(B, max_len - feat.size(1), feat.size(2))
                padded_features.append(torch.cat([feat, padding], dim=1))
            else:
                padded_features.append(feat)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(padded_features, dim=1)  # [B, sum(T'), hidden_dim]
        
        # Initialize output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)  # [B, num_tokens, hidden_dim]
        
        # Apply fusion attention
        fused_features = output_tokens + self.fusion(
            self.fusion_norm(output_tokens),
            multi_scale_features,
            multi_scale_features,
            need_weights=False
        )[0]
        
        # Apply cross-modal fusion with text if available
        if text_features is not None:
            for layer in self.cross_modal_layers:
                fused_features = layer(
                    fused_features, 
                    text_features, 
                    text_attention_mask=text_mask
                )
            
            # Apply contrastive alignment if in training mode
            if self.training:
                # Project features to alignment space
                visual_align = self.align_visual(fused_features)  # [B, num_tokens, hidden_dim]
                text_align = self.align_text(text_features)  # [B, L, hidden_dim]
                
                # Compute similarity matrix
                visual_align = F.normalize(visual_align, dim=-1)
                text_align = F.normalize(text_align, dim=-1)
                
                # Store similarity for loss computation (to be used outside this module)
                self.last_similarity = torch.bmm(visual_align, text_align.transpose(1, 2)) / self.contrastive_temp
        
        # Project to output dimension
        output = self.output_proj(fused_features)
        
        return output


class TextAwareMultiScaleLayer(nn.Module):
    """Cross-modal fusion layer for multi-scale features"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to text
        self.text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, text, text_attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        
        # Cross-attention to text
        if text is not None:
            residual = x
            x = self.norm2(x)
            x = residual + self.dropout(
                self.text_attn(
                    query=x,
                    key=text,
                    value=text,
                    key_padding_mask=None if text_attention_mask is None else ~text_attention_mask.bool(),
                    need_weights=False
                )[0]
            )
        
        # Feed-forward network
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x


class TextAwarePEVLAdapter(BaseProjector):
    """
    Text-aware version of PEVLAdapter that conditions bottleneck
    adapters on instruction text
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        bottleneck_dim: int = 256,
        num_adapter_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_output_tokens: int = 32
    ):
        super().__init__(input_dim, output_dim)
        
        # Text encoder components - BERT embedding layer for text tokens
        from transformers import BertConfig, BertModel
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased', config=bert_config).embeddings.word_embeddings
        
        # Project text features to match input dimension
        self.text_proj = nn.Linear(768, input_dim)  # BERT hidden size is 768
        
        # Create text-conditioned adapters
        self.adapters = nn.ModuleList([
            TextConditionedAdapter(
                d_model=input_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout
            )
            for _ in range(num_adapter_layers)
        ])
        
        # Cross-attention layers for text conditioning
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_adapter_layers)
        ])
        
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_adapter_layers)
        ])
        
        # Output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, input_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Final projection
        self.output_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Process text if available
        text_features = None
        if text_tokens is not None:
            # Get text embeddings using BERT
            with torch.no_grad():
                text_embeds = self.text_embeddings(text_tokens)
            
            # Project to input dimension
            text_features = self.text_proj(text_embeds)
        
        # Apply adapters with text conditioning
        for i, (adapter, cross_attn, cross_norm) in enumerate(zip(self.adapters, self.cross_attn_layers, self.cross_attn_norms)):
            # Apply adapter
            x = adapter(x)
            
            # Apply cross-attention to text if available
            if text_features is not None:
                x_norm = cross_norm(x)
                x = x + cross_attn(
                    query=x_norm,
                    key=text_features,
                    value=text_features,
                    key_padding_mask=None if text_mask is None else ~text_mask.bool(),
                    need_weights=False
                )[0]
        
        # Generate output representation using output tokens
        output_tokens = self.output_tokens.expand(B, -1, -1)
        
        # Cross-attention from output tokens to adapted features
        for cross_attn, cross_norm in zip(self.cross_attn_layers, self.cross_attn_norms):
            query = cross_norm(output_tokens)
            output_tokens = output_tokens + cross_attn(
                query=query,
                key=x,
                value=x,
                need_weights=False
            )[0]
            
            # Also attend to text if available
            if text_features is not None:
                query = cross_norm(output_tokens)
                output_tokens = output_tokens + cross_attn(
                    query=query,
                    key=text_features,
                    value=text_features,
                    key_padding_mask=None if text_mask is None else ~text_mask.bool(),
                    need_weights=False
                )[0]
        
        # Final projection
        output = self.output_proj(output_tokens)
        
        return output


class TextConditionedAdapter(nn.Module):
    """Bottleneck adapter that can be conditioned on text"""
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        
        # Down-projection
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        
        # Activation
        self.act = nn.GELU()
        
        # Up-projection
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply adapter with residual connection
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + x


def get_projector(projector_name, input_dim, output_dim, **kwargs):
    """
    Factory function to create a projector by name
    
    Args:
        projector_name: String identifier for projector type
        input_dim: Input dimension
        output_dim: Output dimension 
        **kwargs: Additional arguments to pass to projector constructor
        
    Returns:
        Initialized projector
    """
    projector_name = projector_name.lower()
    
    # Log the projector creation with its parameters
    logger.info(f"Creating projector: {projector_name} with input_dim={input_dim}, output_dim={output_dim}")
    logger.info(f"Additional kwargs: {kwargs}")
    
    # Basic projectors
    if projector_name == "linear":
        return LinearProjector(input_dim, output_dim)
        
    elif projector_name == "mlp":
        return MLPProjector(
            input_dim, 
            output_dim, 
            hidden_dim=kwargs.get("hidden_dim", input_dim * 2),
            num_layers=kwargs.get("num_layers", 2),
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "qformer":
        return QFormerProjector(
            input_dim, 
            output_dim, 
            num_queries=kwargs.get("num_queries", 32),
            num_layers=kwargs.get("num_layers", 6),  # Use more layers by default for QFormer
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1)
        )
    
    elif projector_name == "blip2_qformer":
        return BLIP2QFormer(
            input_dim,
            output_dim,
            num_queries=kwargs.get("num_queries", 32),
            num_layers=kwargs.get("num_layers", 6),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            intermediate_size=kwargs.get("intermediate_size", 3072),
            use_bert_config=kwargs.get("use_bert_config", True)
        )
    
    elif projector_name == "cross_attention":
        return CrossAttentionProjector(
            input_dim, 
            output_dim, 
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "perceiver":
        return PerceiverProjector(
            input_dim, 
            output_dim, 
            num_latents=kwargs.get("num_latents", 32),
            num_layers=kwargs.get("num_layers", 3),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "adaptive_query":
        return AdaptiveQueryProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
            num_queries=kwargs.get("num_queries", 32)
        )
    
    elif projector_name == "fusion_refinement":
        return FusionRefinementProjector(
            input_dim,
            output_dim,
            num_stages=kwargs.get("num_stages", 3),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "gated_cross_attention":
        return GatedCrossAttentionProjector(
            input_dim,
            output_dim,
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            use_gate_mlp=kwargs.get("use_gate_mlp", True)
        )
        
    elif projector_name == "hierarchical_moe":
        return HierarchicalMoEProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 2048),
            num_experts=kwargs.get("num_experts", 8),
            k=kwargs.get("k", 2),
            num_layers=kwargs.get("num_layers", 2)
        )
        
    elif projector_name == "enhanced_qformer":
        return EnhancedQFormerProjector(
            input_dim,
            output_dim,
            num_queries=kwargs.get("num_queries", 32),
            num_layers=kwargs.get("num_layers", 6),
            num_heads=kwargs.get("num_heads", 8),
            intermediate_dim=kwargs.get("intermediate_dim", 3072),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "multiscale_contrastive":
        return MultiScaleContrastiveProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 2048),
            num_scales=kwargs.get("num_scales", 3),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            contrastive_temp=kwargs.get("contrastive_temp", 0.07)
        )
        
    elif projector_name == "pevl_adapter":
        return PEVLAdapter(
            input_dim,
            output_dim,
            bottleneck_dim=kwargs.get("bottleneck_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
            num_tasks=kwargs.get("num_tasks", 1)
        )
        
    elif projector_name == "self_aggregating_linear":
        return SelfAggregatingLinearProjector(
            input_dim,
            output_dim,
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    elif projector_name == "self_aggregating_mlp":
        return SelfAggregatingMLPProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", input_dim * 2),
            num_layers=kwargs.get("num_layers", 2),
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.1),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
    
    # Text-Aware projectors
    elif projector_name == "text_aware_cross_attention":
        return TextAwareCrossAttentionProjector(
            input_dim,
            output_dim,
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            num_layers=kwargs.get("num_layers", 2),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    elif projector_name == "text_aware_perceiver":
        return TextAwarePerceiverProjector(
            input_dim,
            output_dim,
            num_latents=kwargs.get("num_latents", 32),
            num_latent_dim=kwargs.get("num_latent_dim", 512),
            num_layers=kwargs.get("num_layers", 4),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1)
        )
        
    elif projector_name == "text_aware_adaptive_query":
        return TextAwareAdaptiveQueryProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
            num_queries=kwargs.get("num_queries", 32)
        )
        
    elif projector_name == "text_aware_hierarchical_moe":
        return TextAwareHierarchicalMoEProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 2048),
            num_experts=kwargs.get("num_experts", 8),
            k=kwargs.get("k", 2),
            num_layers=kwargs.get("num_layers", 2),
            num_output_tokens=kwargs.get("num_output_tokens", 32),
            dropout=kwargs.get("dropout", 0.1),
            num_heads=kwargs.get("num_heads", 8)
        )
        
    elif projector_name == "text_aware_fusion_refinement":
        return TextAwareFusionRefinementProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_heads=kwargs.get("num_heads", 8),
            num_refinement_steps=kwargs.get("num_refinement_steps", 3),
            dropout=kwargs.get("dropout", 0.1),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    elif projector_name == "text_aware_gated_cross_attention":
        return TextAwareGatedCrossAttentionProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            num_layers=kwargs.get("num_layers", 3),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    elif projector_name == "text_aware_multiscale_contrastive":
        return TextAwareMultiScaleContrastiveProjector(
            input_dim,
            output_dim,
            hidden_dim=kwargs.get("hidden_dim", 2048),
            num_scales=kwargs.get("num_scales", 3),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.1),
            contrastive_temp=kwargs.get("contrastive_temp", 0.07),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    elif projector_name == "text_aware_pevl_adapter":
        return TextAwarePEVLAdapter(
            input_dim,
            output_dim,
            bottleneck_dim=kwargs.get("bottleneck_dim", 256),
            num_adapter_layers=kwargs.get("num_adapter_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
            num_heads=kwargs.get("num_heads", 8),
            num_output_tokens=kwargs.get("num_output_tokens", 32)
        )
        
    else:
        raise ValueError(f"Unknown projector type: {projector_name}")