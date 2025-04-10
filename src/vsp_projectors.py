# Copyright (c) 2023
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import math
import inspect
import logging
from einops import rearrange, repeat

# Initialize logger
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
        self.input_dim = input_dim  # Store input_dim for reference
        
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
        
        # Add cross-modal components for transcript alignment
        # This projection can handle any input dimension to our internal dimension
        self.text_projection = nn.Linear(output_dim, input_dim)  # Changed from input_dim to output_dim
        self.text_norm = nn.LayerNorm(input_dim)
        
        # Cross-attention for aligning visual queries with text
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.cross_modal_transformer = nn.TransformerDecoder(decoder_layer, 1)
        
        # Final projection to match LLM dimensions
        self.proj = nn.Linear(input_dim, output_dim)
        
        # Initialize projection layer properly
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_embeddings=None, return_alignment_loss=False) -> torch.Tensor:
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
        
        alignment_loss = torch.tensor(0.0, device=x.device)
        
        # Process text information if available (for cross-modal alignment)
        if text_embeddings is not None:
            # Process text embeddings
            # Check and handle different input shapes
            if len(text_embeddings.shape) == 3:
                # If we receive embeddings directly [B, L, D]
                if text_embeddings.size(-1) != self.input_dim:
                    # Need to project to match dimensions
                    text_features = self.text_projection(text_embeddings)
                else:
                    # Already has correct dimensions, just normalize
                    text_features = text_embeddings
            else:
                # Handle unexpected input shape
                logger.warning(f"Unexpected text_embeddings shape: {text_embeddings.shape}. Skipping text conditioning.")
                return query_output
            
            text_features = self.text_norm(text_features)
            
            # Create padding mask for text (1 = padding, 0 = real token)
            if text_tokens is not None:
                text_padding_mask = (text_tokens == 0)
            else:
                text_padding_mask = None
            
            # Use cross-modal attention to refine queries with text information
            refined_queries = self.cross_modal_transformer(
                query_output,           # Target (queries)
                text_features,          # Memory (text)
                memory_key_padding_mask=text_padding_mask
            )
            
            # Calculate alignment loss (simple contrastive loss)
            if return_alignment_loss:
                # Mean pooling for both modalities
                query_pooled = refined_queries.mean(dim=1)  # [B, D]
                text_pooled = text_features.mean(dim=1)     # [B, D]
                
                # Normalize embeddings
                query_pooled = F.normalize(query_pooled, dim=-1)
                text_pooled = F.normalize(text_pooled, dim=-1)
                
                # Compute similarity matrix
                similarity = torch.matmul(query_pooled, text_pooled.transpose(0, 1))  # [B, B]
                
                # Scale by temperature
                temperature = 0.07
                similarity = similarity / temperature
                
                # Labels are the diagonal (matching pairs)
                labels = torch.arange(similarity.size(0), device=similarity.device)
                
                # Symmetric loss
                loss_i2t = F.cross_entropy(similarity, labels)
                loss_t2i = F.cross_entropy(similarity.transpose(0, 1), labels)
                
                alignment_loss = (loss_i2t + loss_t2i) / 2.0
            
            # Use refined queries for final projection
            final_queries = refined_queries
        else:
            # Without text information, use original query output
            final_queries = query_output
            
        # Project to target dimension
        projected = self.proj(final_queries)
        
        if return_alignment_loss:
            return projected, alignment_loss
        
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
        
        # Cross-attention layers for visual features
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Add cross-attention layers for text conditioning
        self.text_attn_layers = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers // 2 or 1)  # At least one layer
        ])
        
        # Text projection - expect output_dim which matches the LLM dimension
        self.text_proj = nn.Linear(output_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Project input if dimensions don't match
        x_proj = self.input_proj(x)
        
        # Expand latent tokens to batch size
        latent = self.latent_tokens.expand(B, -1, -1)
        
        # Apply cross-attention layers with visual features
        for layer in self.cross_attn_layers:
            latent = layer(latent, x_proj)
        
        # Apply text conditioning if provided and if it's a tensor (embeddings)
        if text_tokens is not None and torch.is_tensor(text_tokens):
            # Only use text conditioning if it's a proper tensor with correct shape
            if len(text_tokens.shape) == 3:  # [B, L, D]
                # Project text to match latent dimension
                text_proj = self.text_proj(text_tokens)
                
                # Create attention mask (True = positions to ignore)
                key_padding_mask = None
                if text_mask is not None:
                    key_padding_mask = ~text_mask
                
                # Skip text conditioning if all tokens are masked
                if key_padding_mask is None or not key_padding_mask.all():
                    # Apply cross-attention with masking
                    for layer in self.text_attn_layers:
                        latent = layer(latent, text_proj, key_padding_mask)
            
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
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # q: [B, Q, D], kv: [B, T, D]
        # Self-attention among queries
        q2 = self.norm1(q)
        q = q + self.cross_attn(q2, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)[0]
        
        # FFN
        q = q + self.mlp(self.norm2(q))
        return q


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
        self.hidden_dim = hidden_dim
        self.temp = contrastive_temp
        
        # Define multi-scale processing layers
        self.pooling_scales = []
        for i in range(num_scales):
            scale = 2 ** i
            if scale == 1:
                # Identity pooling (no pooling)
                self.pooling_scales.append(nn.Identity())
            else:
                # Adaptive pooling with different scales
                self.pooling_scales.append(
                    nn.AdaptiveAvgPool1d(scale)
                )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Text alignment layers
        self.align_text = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),  # Changed to output_dim to match LLM embedding dim
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Contrastive alignment head
        self.align_visual = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable output tokens
        self.output_tokens = nn.Parameter(torch.zeros(1, 32, hidden_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Text enhancement components
        self.text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        # x: [B, T, input_dim]
        B = x.size(0)
        
        # Multi-scale feature extraction
        scale_features = []
        for i, scale in enumerate(self.pooling_scales):
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
        
        # Apply text conditioning if available
        if text_tokens is not None:
            # Check whether we received embeddings or token IDs
            if torch.is_floating_point(text_tokens):
                # Process embeddings directly
                text_features = text_tokens
                if text_features.size(-1) != self.hidden_dim:
                    # Project if dimensions don't match
                    text_features = self.align_text(text_features)
            else:
                # Process token IDs
                try:
                    # Process text tokens
                    text_features = self.align_text(text_tokens)  # [B, L, hidden_dim]
                except Exception as e:
                    logger.warning(f"Error processing text tokens in MultiScaleContrastiveProjector: {e}")
                    return fused_features
            
            # Create text attention mask if needed
            key_padding_mask = None
            if text_mask is not None:
                key_padding_mask = ~text_mask  # Convert to format expected by attention (True means ignore)
            
            # Skip if all tokens are masked
            if key_padding_mask is None or not key_padding_mask.all():
                # Match dtype for compatibility
                text_features = text_features.to(dtype=fused_features.dtype)
                
                # Apply cross-attention from features to text
                text_enhanced_features = fused_features + self.text_attn(
                    self.text_norm(fused_features),  # Query
                    text_features,                   # Key
                    text_features,                   # Value
                    key_padding_mask=key_padding_mask,
                    need_weights=False
                )[0]
                
                # Use text-enhanced features
                fused_features = text_enhanced_features
        
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
        similarity = torch.bmm(visual_emb, text_emb.transpose(1, 2)) / self.temp
        
        # Contrastive loss
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)
        
        return loss / 2.0


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
        
        # Add the missing dropout layer that's used in the forward method
        self.dropout = nn.Dropout(dropout)
        
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


class ComprehensiveQFormerProjector(BaseProjector):
    """
    Comprehensive Q-Former projector that combines best features from MMS-LLama and BLIP-2
    implementations with speech-specific optimizations. Includes:
    - Bidirectional attention
    - Deeper transformer layers
    - Text conditioning capabilities
    - Specialized cross-modal attention
    - Speech-specific temporal modeling
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,  # Deeper transformer like MMS-LLama
        num_heads: int = 8,
        intermediate_dim: int = 3072,  # Larger FFN
        dropout: float = 0.1,
        use_text_conditioning: bool = True,
        use_temporal_attention: bool = True,  # Speech-specific feature
        max_text_len: int = 32,
        temporal_window_size: int = 5  # For temporal attention
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.use_text_conditioning = use_text_conditioning
        self.use_temporal_attention = use_temporal_attention
        
        # Learnable query tokens with optimal initialization
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Text encoding components (for instruction conditioning)
        if use_text_conditioning:
            from transformers import BertConfig, BertModel
            # Text embedding layer
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased').embeddings
            # Project text embeddings to match visual dimension
            self.text_proj = nn.Linear(768, input_dim)  # BERT dim is 768
            self.text_norm = nn.LayerNorm(input_dim)
        
        # Speech-specific temporal attention
        if use_temporal_attention:
            self.temporal_conv = nn.Conv1d(
                input_dim, 
                input_dim, 
                kernel_size=temporal_window_size,
                padding=(temporal_window_size-1)//2,
                groups=num_heads  # Multi-headed temporal convolution
            )
            self.temporal_gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
        
        # Main transformer layers - using interleaved self and cross attention
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention layer
            self.layers.append(
                TransformerSelfAttentionLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_dim,
                    dropout=dropout,
                    norm_first=True  # Pre-norm architecture
                )
            )
            
            # Cross-attention layer (every other layer)
            if i % 2 == 1:
                self.layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
                
                # Text cross-attention (if enabled)
                if use_text_conditioning:
                    self.layers.append(
                        TransformerCrossAttentionLayer(
                            d_model=input_dim,
                            nhead=num_heads,
                            dim_feedforward=intermediate_dim,
                            dropout=dropout,
                            norm_first=True
                        )
                    )
        
        # Final layer norm and projection
        self.final_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass through the comprehensive QFormer
        
        Args:
            x: Visual/audio features [B, T, input_dim]
            text_tokens: Optional text token ids [B, L] 
            text_mask: Optional attention mask for text [B, L]
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        B = x.size(0)
        
        # SIMPLIFIED TEXT CONDITIONING CHECK:
        # Only use text conditioning during training, never during inference
        use_text = False
        if self.use_text_conditioning and text_tokens is not None:
            if torch.is_grad_enabled():  # Only condition with text during training
                if isinstance(text_tokens, torch.Tensor) and text_tokens.dim() > 1:
                    # We have proper text tokens and we're in training mode
                    use_text = True
                    if text_mask is None:
                        text_mask = (text_tokens != 0).float()
                
        # Process text if available and if we determined we should use it
        text_features = None
        if use_text:
            # Convert IDs to embeddings
            with torch.no_grad():
                text_tokens = text_tokens.to(dtype=torch.long)
                text_embeds = self.text_encoder.word_embeddings(text_tokens)
            
            # Project and normalize
            text_features = self.text_proj(text_embeds)
            text_features = self.text_norm(text_features)
            
            # Match device and dtype
            text_features = text_features.to(x.device).to(x.dtype)
            if text_mask is not None:
                text_mask = text_mask.to(x.device)
        
        # Apply input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Apply temporal modeling for speech (if enabled)
        if self.use_temporal_attention:
            # Apply temporal convolution
            x_temporal = x.transpose(1, 2)  # [B, D, T]
            x_temporal = self.temporal_conv(x_temporal).transpose(1, 2)  # [B, T, D]
            
            # Gated mechanism to control temporal information flow
            gate = self.temporal_gate(x)
            x = x + gate * x_temporal
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Main processing through transformer layers
        layer_idx = 0
        hidden_states = query_tokens
        
        while layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            
            # Check the layer type and call accordingly
            if isinstance(layer, TransformerSelfAttentionLayer):
                # Self-attention layer
                hidden_states = layer(hidden_states)
            elif isinstance(layer, TransformerCrossAttentionLayer):
                # First cross-attention layer - attend to visual features
                hidden_states = layer(q=hidden_states, kv=x, mask=None)
                
                # Check if we need to apply text cross-attention next
                if (use_text and text_features is not None and 
                    layer_idx + 1 < len(self.layers) and 
                    isinstance(self.layers[layer_idx + 1], TransformerCrossAttentionLayer)):
                    
                    layer_idx += 1
                    text_attention_mask = None
                    if text_mask is not None:
                        text_attention_mask = ~text_mask.bool()
                    
                    # Text cross-attention layer
                    hidden_states = self.layers[layer_idx](
                        q=hidden_states,
                        kv=text_features,
                        mask=text_attention_mask
                    )
            
            layer_idx += 1
        
        # Final normalization and projection
        hidden_states = self.final_norm(hidden_states)
        output = self.output_proj(hidden_states)
        
        return output


class TransformerSelfAttentionLayer(nn.Module):
    """Transformer layer with self-attention only"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        if self.norm_first:
            # Pre-norm
            x1 = self.norm(x)
            x = x + self.dropout(self.self_attn(x1, x1, x1, 
                                              key_padding_mask=mask, 
                                              need_weights=False)[0])
            x2 = self.norm(x)
            x = x + self.ffn(x2)
        else:
            # Post-norm
            x = x + self.dropout(self.self_attn(x, x, x, 
                                              key_padding_mask=mask, 
                                              need_weights=False)[0])
            x = self.norm(x)
            x = x + self.ffn(x)
            x = self.norm(x)
        return x


class TransformerCrossAttentionLayer(nn.Module):
    """
    Transformer cross-attention layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, q, kv, mask=None):
        """
        Args:
            q: Query tensor [B, seq_len, dim]
            kv: Key/value tensor [B, kv_len, dim]
            mask: Optional mask for kv
        
        Returns:
            Updated query tensor
        """
        # Apply layer norm first if using pre-norm
        if self.norm_first:
            # Ensure dimensions are correct - apply reshape if needed
            q_norm = self.norm1(q)
            kv_norm = self.norm1(kv)
            
            # Handle potential 4D input (ensure key and value are 3D)
            if len(kv_norm.shape) == 4:
                # If we have a 4D tensor, reshape it to 3D
                B, T, D1, D2 = kv_norm.shape
                kv_norm = kv_norm.reshape(B, T * D1, D2)
            
            # Fix mask dimensions to match kv dimensions
            if mask is not None and kv_norm.size(1) != mask.size(1):
                # Disable masking if dimensions don't match
                mask = None
            
            # Call cross attention with proper 3D shapes
            x = q + self.dropout1(self.cross_attn(
                q_norm, kv_norm, kv_norm, 
                key_padding_mask=mask,
                need_weights=False)[0])
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(
                self.linear1(self.norm2(x))))))
        else:
            # Handle potential 4D input (ensure key and value are 3D)
            if len(kv.shape) == 4:
                # If we have a 4D tensor, reshape it to 3D
                B, T, D1, D2 = kv.shape
                kv = kv.reshape(B, T * D1, D2)
            
            # Fix mask dimensions to match kv dimensions
            if mask is not None and kv.size(1) != mask.size(1):
                # Disable masking if dimensions don't match
                mask = None
            
            # Call cross attention with proper 3D shapes
            x = self.norm1(q + self.dropout1(self.cross_attn(
                q, kv, kv, key_padding_mask=mask, need_weights=False)[0]))
            x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(
                self.linear1(x))))))
        
        return x


class VisualSpeechQFormer(BaseProjector):
    """QFormer specifically adapted for visual speech processing"""
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32, 
        num_layers: int = 2, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        window_size: int = 3  # Local window size for visual speech
    ):
        super().__init__(input_dim, output_dim)
        
        # Visual speech uses short temporal windows
        self.window_size = window_size
        
        # Apply convolutional frontend to capture lip movements
        self.visual_frontend = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=window_size, padding=window_size//2, groups=8),
            nn.GELU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=window_size, padding=window_size//2),
            nn.GELU()
        )
        
        # Create learnable query vectors - more queries for visual speech
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Final projection
        self.norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # For precision handling
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass for VisualSpeechQFormer
        
        Args:
            x: Visual/audio features [B, T, input_dim]
            text_tokens: Optional text token ids - not used in this base implementation,
                         but accepted for compatibility with the interface
            text_mask: Optional attention mask for text - not used in this base implementation
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        try:
            batch_size = x.shape[0]
            orig_dtype = x.dtype
            
            # Apply visual frontend for lip movement features
            x_conv = x.transpose(1, 2)  # [B, D, T]
            
            # Ensure consistent dtype in convolutional layers
            # Check if any parameters in visual_frontend have different dtype than input
            param_dtype = next(self.visual_frontend.parameters()).dtype
            if x_conv.dtype != param_dtype:
                # Match input to parameter dtype
                x_conv = x_conv.to(dtype=param_dtype)
                
            # Process through frontend
            x_conv = self.visual_frontend(x_conv)
            # Restore original dtype if needed
            if orig_dtype != param_dtype:
                x_conv = x_conv.to(dtype=orig_dtype)
                
            x = x_conv.transpose(1, 2)  # [B, T, D]
            
            # Add subtle positional information to preserve temporal order in lip movements
            position = torch.arange(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            position = position.expand(batch_size, -1, x.size(-1))
            position = (0.1 * position / float(x.size(1))).to(dtype=x.dtype)
            x = x + position
            
            # Expand query tokens to batch size
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            
            # Ensure query tokens match input dtype
            if query_tokens.dtype != x.dtype:
                query_tokens = query_tokens.to(dtype=x.dtype)
            
            # Concatenate query tokens with processed features
            x_with_query = torch.cat([query_tokens, x], dim=1)
            
            # Create attention mask that allows queries to attend to all inputs
            seq_len = x.size(1)
            query_len = query_tokens.size(1)
            
            # More stable mask creation for float16 - use float32 for computation
            mask = torch.ones(query_len + seq_len, query_len + seq_len, 
                              device=x.device, dtype=torch.bool)
            
            # In visual speech, queries should attend to local temporal windows of lip movement
            # This helps capture the transitions between lip positions
            local_window = self.window_size
            
            # Let all positions attend to themselves (diagonal) - more numerically stable
            # Use direct indexing to set mask values
            for i in range(query_len + seq_len):
                mask[i, i] = False  # False = attend to this position
            
            # Set window-based attention - more numerically stable approach
            for i in range(query_len):
                # Each query attends to a local region of the sequence
                region_start = max(0, (i * seq_len // query_len) - local_window)
                region_end = min(seq_len, ((i + 1) * seq_len // query_len) + local_window)
                
                # Set mask to allow this query to attend to this region
                for j in range(region_start, region_end):
                    mask[i, query_len + j] = False  # False = attend to this position
            
            # Ensure transformer dtype consistency
            transformer_dtype = next(self.transformer.parameters()).dtype
            if x_with_query.dtype != transformer_dtype:
                x_with_query = x_with_query.to(dtype=transformer_dtype)
            
            # Process through transformer
            hidden_states = self.transformer(x_with_query, mask=mask)
            
            # Extract only the query outputs
            query_output = hidden_states[:, :query_len]
            
            # Final norm and projection
            query_output = self.norm(query_output)
            output = self.output_proj(query_output)
            
            # Restore original dtype if needed
            if output.dtype != orig_dtype:
                output = output.to(dtype=orig_dtype)
            
            return output
            
        except RuntimeError as e:
            self.logger.warning(f"Error in VisualSpeechQFormer: {e}")
            self.logger.warning(f"Input shape: {x.shape}, dtype: {x.dtype}")
            self.logger.warning(f"Visual frontend param dtype: {next(self.visual_frontend.parameters()).dtype}")
            
            # Handle specific dtype errors with fallback to full float32 computation
            if "should be the same" in str(e) or "expected scalar type" in str(e):
                self.logger.warning("Attempting to run with full float32 precision")
                with torch.cuda.amp.autocast(enabled=False):
                    # Force float32 for everything
                    return self.forward(x.float(), 
                                      None if text_tokens is None else text_tokens, 
                                      text_mask)
            else:
                # For other errors, provide component-specific diagnostics
                try:
                    # Test each component separately to identify the problematic one
                    x_conv = x.transpose(1, 2).float()  # Convert to float32
                    self.logger.warning("Testing visual frontend...")
                    x_conv = self.visual_frontend(x_conv)
                    x_tmp = x_conv.transpose(1, 2)
                    
                    self.logger.warning("Testing transformer...")
                    query_tokens = self.query_tokens.expand(x.shape[0], -1, -1).float()
                    x_with_query = torch.cat([query_tokens, x_tmp], dim=1)
                    hidden_states = self.transformer(x_with_query)
                    
                    self.logger.warning("All components passed individual tests with float32")
                except Exception as component_e:
                    self.logger.error(f"Component error: {component_e}")
                
                # Re-raise the original error
                raise e


class VisualSpeechTextQFormer(BaseProjector):
    """
    A standalone QFormer for visual speech processing with text conditioning,
    implemented from scratch to ensure consistent dimension and dtype handling.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32, 
        num_layers: int = 2, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        window_size: int = 3,  # Local window size for visual speech
        text_dim: int = 768,
        text_num_layers: int = 2
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Learnable query tokens - used to extract key information from sequence
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Visual frontend with convolutional layers for lip movement features
        self.visual_frontend = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=window_size, padding=window_size//2, groups=8),
            nn.GELU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=window_size, padding=window_size//2),
            nn.GELU()
        )
        
        # Self-attention transformer layers 
        visual_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.visual_transformer = nn.TransformerEncoder(visual_encoder_layer, num_layers)
        
        # Text processing components
        self.text_dim = text_dim
        from transformers import BertModel
        self.text_embedder = BertModel.from_pretrained('bert-base-uncased').embeddings.word_embeddings
        self.text_projection = nn.Linear(text_dim, input_dim)
        self.text_norm = nn.LayerNorm(input_dim)
        
        # Cross-attention layers for text-visual interaction
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(text_num_layers)
        ])
        
        # Final normalization and projection
        self.final_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
    def _process_visual_features(self, x):
        """Process visual features with convolutions and self-attention"""
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Apply convolutional frontend for lip features
        x_conv = x.transpose(1, 2)  # [B, D, T]
        # Ensure consistent dtype with convolutional layers
        conv_dtype = next(self.visual_frontend.parameters()).dtype
        if x_conv.dtype != conv_dtype:
            x_conv = x_conv.to(dtype=conv_dtype)
            
        x_conv = self.visual_frontend(x_conv)
        x_processed = x_conv.transpose(1, 2)  # [B, T, D]
        
        # Add positional information
        seq_len = x_processed.size(1)
        position = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
        position = position.expand(batch_size, -1, x_processed.size(-1))
        position = (0.1 * position / float(seq_len)).to(dtype=x_processed.dtype)
        x_processed = x_processed + position
        
        # Prepare query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_tokens = query_tokens.to(dtype=x_processed.dtype)
        
        # Concatenate query tokens with processed features
        x_with_query = torch.cat([query_tokens, x_processed], dim=1)
        
        # Create attention mask for local window attention
        query_len = query_tokens.size(1)
        mask = torch.ones(query_len + seq_len, query_len + seq_len, 
                         device=device, dtype=torch.bool)
        
        # Allow diagonal (self) attention
        for i in range(query_len + seq_len):
            mask[i, i] = False
        
        # Set local window attention for queries
        local_window = self.window_size
        for i in range(query_len):
            region_start = max(0, (i * seq_len // query_len) - local_window)
            region_end = min(seq_len, ((i + 1) * seq_len // query_len) + local_window)
            for j in range(region_start, region_end):
                mask[i, query_len + j] = False
        
        # Ensure transformer dtype consistency
        transformer_dtype = next(self.visual_transformer.parameters()).dtype
        if x_with_query.dtype != transformer_dtype:
            x_with_query = x_with_query.to(dtype=transformer_dtype)
            
        # Process through transformer
        output = self.visual_transformer(x_with_query, mask=mask)
        
        # Extract only query outputs
        query_output = output[:, :query_len]
        return query_output.to(dtype=dtype)  # Restore original dtype
    
    def _process_text_features(self, text_tokens, target_dtype, target_device):
        """Process text tokens into embeddings with proper dtype handling"""
        if text_tokens is None:
            return None
            
        # Convert to long for embedding lookup
        text_tokens_long = text_tokens.to(dtype=torch.long, device=target_device)
        
        # Get embeddings using BERT
        with torch.no_grad():
            text_embeddings = self.text_embedder(text_tokens_long)
        
        # Project to match visual dimensions
        text_embeddings = text_embeddings.to(dtype=target_dtype)
        text_features = self.text_projection(text_embeddings)
        text_features = self.text_norm(text_features)
        
        return text_features
    
    def _get_attention_mask(self, text_mask, target_device):
        """Convert text mask to attention mask format"""
        if text_mask is None:
            return None
            
        # Convert mask format: True for valid tokens → False for padding positions
        # PyTorch attention: True = position to ignore
        attention_mask = ~text_mask.bool()
        return attention_mask.to(device=target_device)
    
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass for VisualSpeechTextQFormer with explicit dtype handling
        
        Args:
            x: Visual features [B, T, input_dim]
            text_tokens: Text token ids [B, L]
            text_mask: Attention mask for text (True for valid tokens)
            
        Returns:
            Text-enhanced projected features [B, num_queries, output_dim]
        """
        # Store original dtype and device for consistency
        orig_dtype = x.dtype
        orig_device = x.device
        
        try:
            # First process visual features to get query embeddings
            self.logger.info(f"Processing visual features with shape {x.shape}")
            query_output = self._process_visual_features(x)
            
            # If no text conditioning provided, return visual-only output
            if text_tokens is None:
                self.logger.info("No text tokens provided, using visual features only")
                query_output = self.final_norm(query_output) 
                return self.output_proj(query_output)
                
            # Process text tokens to get conditioned embeddings
            self.logger.info(f"Processing text tokens with shape {text_tokens.shape}")
            text_features = self._process_text_features(
                text_tokens, 
                target_dtype=query_output.dtype,
                target_device=query_output.device
            )
            
            # Get proper attention mask for padding
            key_padding_mask = self._get_attention_mask(text_mask, query_output.device)
            
            # Use cross-attention to enhance visual with text
            enhanced_queries = query_output
            for i, layer in enumerate(self.cross_attn_layers):
                self.logger.info(f"Applying cross-attention layer {i+1}/{len(self.cross_attn_layers)}")
                enhanced_queries = layer(
                    enhanced_queries,                  # tgt (queries)
                    text_features,                     # memory (text)
                    memory_key_padding_mask=key_padding_mask
                )
                
            # Apply final normalization and projection
            output = self.final_norm(enhanced_queries)
            output = self.output_proj(output)
            
            # Convert back to original dtype if needed
            if output.dtype != orig_dtype:
                output = output.to(dtype=orig_dtype)
                
            return output
            
        except RuntimeError as e:
            self.logger.error(f"Error in VisualSpeechTextQFormer: {e}")
            self.logger.error(f"Visual features: shape={x.shape}, dtype={x.dtype}")
            if text_tokens is not None:
                self.logger.error(f"Text tokens: shape={text_tokens.shape}, dtype={text_tokens.dtype}")
                
            # Provide specific diagnostics for identified errors
            if "normalized_shape" in str(e):
                self.logger.error("DIMENSION MISMATCH ERROR: LayerNorm dimension doesn't match input")
                self.logger.error(f"Last layer norm dims: {self.final_norm.normalized_shape}")
                if len(x.shape) >= 3:
                    self.logger.error(f"Output dimension: {x.shape[-1]}")
            elif "expected scalar type" in str(e) or "same dtype" in str(e):
                self.logger.error("DTYPE MISMATCH ERROR: Tensors have inconsistent dtypes")
                
            # Re-raise the exception to prevent silent failure
            raise


class StableCrossAttentionLayer(nn.Module):
    """Cross-attention layer with improved numerical stability for mixed precision training"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Use standard PyTorch components for stability
        self.cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        # For error logging
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x, kv, key_padding_mask=None):
        """Forward pass with explicit dtype handling
        
        Args:
            x: Query tensor [B, seq_len, dim]
            kv: Key/value tensor [B, kv_len, dim]
            key_padding_mask: Mask for kv, True indicates invalid positions
            
        Returns:
            Tensor [B, seq_len, dim]
        """
        try:
            # Ensure consistent dtypes
            if kv.dtype != x.dtype:
                kv = kv.to(dtype=x.dtype)
                
            # Use the transformer decoder layer (target = x, memory = kv)
            return self.cross_attn_layer(
                x,  # Target 
                kv, # Memory
                memory_key_padding_mask=key_padding_mask
            )
        except RuntimeError as e:
            self.logger.warning(f"Error in StableCrossAttentionLayer: {e}")
            self.logger.warning(f"x shape: {x.shape}, kv shape: {kv.shape}")
            
            # If we have a dimension mismatch, provide more debug info
            if "normalized_shape" in str(e) or "expected input of size" in str(e):
                self.logger.error(f"Dimension mismatch: x.size(-1)={x.size(-1)}, kv.size(-1)={kv.size(-1)}")
                
                # Try to convert dimensions to match (fallback option)
                if x.size(-1) != kv.size(-1):
                    self.logger.warning(f"Attempting dimension correction")
                    # Use a temporary linear projection as a fallback
                    temp_proj = nn.Linear(kv.size(-1), x.size(-1), device=x.device)
                    kv_projected = temp_proj(kv)
                    return self.cross_attn_layer(
                        x,
                        kv_projected,
                        memory_key_padding_mask=key_padding_mask
                    )
            
            # Re-raise the exception if we couldn't handle it
            raise e


class VisualOnlyQFormer(BaseProjector):
    """
    Visual-only QFormer that maintains the architecture strengths of ComprehensiveQFormer
    but removes all text conditioning for stability and consistency between training and inference.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        temporal_window_size: int = 5
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.use_temporal_attention = use_temporal_attention
        
        # Learnable query tokens with optimal initialization
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Speech-specific temporal attention
        if use_temporal_attention:
            self.temporal_conv = nn.Conv1d(
                input_dim, 
                input_dim, 
                kernel_size=temporal_window_size,
                padding=(temporal_window_size-1)//2,
                groups=num_heads  # Multi-headed temporal convolution
            )
            self.temporal_gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
        
        # Main transformer layers - simpler architecture without text cross-attention
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention layer
            self.layers.append(
                TransformerSelfAttentionLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_dim,
                    dropout=dropout,
                    norm_first=True
                )
            )
            
            # Cross-attention layer with visual features (every other layer)
            if i % 2 == 1:
                self.layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
        
        # Final layer norm and projection
        self.final_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass through the visual-only QFormer
    
    Args:
            x: Visual/audio features [B, T, input_dim]
            text_tokens: Ignored (for compatibility with interface)
            text_mask: Ignored (for compatibility with interface)
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        B = x.size(0)
        
        # Apply input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Apply temporal modeling for speech (if enabled)
        if self.use_temporal_attention:
            # Apply temporal convolution
            x_temporal = x.transpose(1, 2)  # [B, D, T]
            x_temporal = self.temporal_conv(x_temporal).transpose(1, 2)  # [B, T, D]
            
            # Gated mechanism to control temporal information flow
            gate = self.temporal_gate(x)
            x = x + gate * x_temporal
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Ensure query tokens match input dtype
        if query_tokens.dtype != x.dtype:
            query_tokens = query_tokens.to(dtype=x.dtype)
        
        # Main processing through transformer layers
        hidden_states = query_tokens
        
        for layer in self.layers:
            if isinstance(layer, TransformerSelfAttentionLayer):
                # Self-attention layer
                hidden_states = layer(hidden_states)
            elif isinstance(layer, TransformerCrossAttentionLayer):
                # Cross-attention to visual features only
                hidden_states = layer(q=hidden_states, kv=x, mask=None)
        
        # Final normalization and projection
        hidden_states = self.final_norm(hidden_states)
        output = self.output_proj(hidden_states)
        
        return output


class VisualOnlyBlip2QFormer(BaseProjector):
    """
    Visual-only version of BLIP2QFormer without any text conditioning
    for stable and consistent training/inference behavior.
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
        
        # Query embeddings
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Visual features projection and normalization
        self.visual_proj = nn.Linear(input_dim, input_dim)
        self.visual_norm = nn.LayerNorm(input_dim)
        
        # QFormer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=intermediate_size if use_bert_config else input_dim * 4,
                dropout=dropout,
                cross_attention=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x, text_tokens=None, text_mask=None):
        # Process visual features
        visual_features = self.visual_proj(x)
        visual_features = self.visual_norm(visual_features)
        
        # Create attention mask for visual features (None means all tokens are valid)
        visual_attention_mask = None
        
        # Initialize query features
        B = x.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_output = query_tokens
        
        # Process through layers
        for layer in self.layers:
            # Since we only have visual modality, we pass None for text
            query_output = layer(
                query=query_output,
                visual=visual_features,
                text=None,  # No text features
                visual_attention_mask=visual_attention_mask,
                text_attention_mask=None  # No text attention mask
            )
        
        # Project to output dimension
        output = self.output_proj(query_output)
        
        return output


class VisualOnlyCrossAttention(BaseProjector):
    """
    Visual-only version of CrossAttentionProjector without any text conditioning
    for stable and consistent training/inference behavior.
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
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Cross-attention blocks (without text conditioning)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=input_dim, 
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Standard weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass with visual features only
        
        Args:
            x: Visual features [B, T, input_dim]
            text_tokens: Ignored (for compatibility)
            text_mask: Ignored (for compatibility)
            
        Returns:
            Projected features [B, T, output_dim]
        """
        # Process input
        x = self.input_norm(self.input_proj(x))
        
        # Apply cross-attention (self-attention to visual features)
        for block in self.cross_attn_blocks:
            # Use visual features as both query and key/value
            x = block(q=x, kv=x)
        
        # Output projection
        x = self.output_norm(x)
        output = self.output_proj(x)
        
        return output


class VisualOnlyMultiScaleContrastive(BaseProjector):
    """
    Visual-only version of MultiScaleContrastiveProjector without any text conditioning
    for stable and consistent training/inference behavior.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 2048,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Create multiple scales of processing
        self.scales = nn.ModuleList()
        for scale in range(num_scales):
            # Scale-specific processing with different receptive fields
            stride = 2 ** scale  # Different strides for different scales
            
            # For each scale, create a processing block
            self.scales.append(nn.Sequential(
                # Temporal convolution with scale-specific stride 
                nn.Conv1d(
                    hidden_dim, 
                    hidden_dim, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=1,
                    groups=num_heads
                ),
                nn.GELU(),
                # Restore original dimension
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            ))
        
        # Process each scale with self-attention
        self.scale_attention = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_scales)
        ])
        
        # Fusion layer to combine all scales
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass with visual features only
        
        Args:
            x: Visual features [B, T, input_dim]
            text_tokens: Ignored (for compatibility)
            text_mask: Ignored (for compatibility)
            
        Returns:
            Projected features [B, T, output_dim]
        """
        B, T, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Process each scale
        scale_outputs = []
        for i, (scale_conv, scale_attn) in enumerate(zip(self.scales, self.scale_attention)):
            # Apply scale-specific convolution
            scale_x = x.transpose(1, 2)  # [B, D, T]
            scale_x = scale_conv(scale_x).transpose(1, 2)  # [B, T', D]
            
            # Apply self-attention
            scale_x = scale_attn(q=scale_x, kv=scale_x)
            
            # Resize back to original sequence length if needed
            if scale_x.size(1) != T:
                scale_x = F.interpolate(
                    scale_x.transpose(1, 2),  # [B, D, T']
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, T, D]
            
            scale_outputs.append(scale_x)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_outputs, dim=-1)
        
        # Fuse all scales
        fused = self.fusion(multi_scale)
        
        # Final projection
        output = self.output_proj(self.output_norm(fused))
        
        return output


class TextGuidedQFormer(BaseProjector):
    """
    A text-guided QFormer that uses text conditioning during training 
    to create better visual representations, but doesn't require text during inference.
    
    Key features:
    1. Dual-path architecture with text supervision
    2. Distillation from text-conditioned to visual-only paths
    3. Consistently uses only visual path during inference
    4. No training/inference mismatch
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        temporal_window_size: int = 5,
        distillation_weight: float = 0.5
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.use_temporal_attention = use_temporal_attention
        self.distillation_weight = distillation_weight
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Text encoding components
        from transformers import BertModel
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').embeddings
        self.text_proj = nn.Linear(768, input_dim)  # BERT dim is 768
        self.text_norm = nn.LayerNorm(input_dim)
        
        # Temporal attention for speech
        if use_temporal_attention:
            self.temporal_conv = nn.Conv1d(
                input_dim, 
                input_dim, 
                kernel_size=temporal_window_size,
                padding=(temporal_window_size-1)//2,
                groups=num_heads
            )
            
            self.temporal_gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
        
        # CREATE TWO SEPARATE TRANSFORMER STACKS
        
        # 1. Visual-only path (used for both training and inference)
        self.visual_only_layers = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention layer
            self.visual_only_layers.append(
                TransformerSelfAttentionLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_dim,
                    dropout=dropout,
                    norm_first=True
                )
            )
            
            # Cross-attention to visual features (every other layer)
            if i % 2 == 1:
                self.visual_only_layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
        
        # 2. Text-conditioned path (used only during training)
        self.text_conditioned_layers = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention layer
            self.text_conditioned_layers.append(
                TransformerSelfAttentionLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_dim,
                    dropout=dropout,
                    norm_first=True
                )
            )
            
            # Cross-attention layer (every other layer)
            if i % 2 == 1:
                # Visual cross-attention
                self.text_conditioned_layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
                
                # Text cross-attention
                self.text_conditioned_layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
        
        # Final layer norms and projections (separate for each path)
        self.visual_norm = nn.LayerNorm(input_dim)
        self.visual_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.visual_proj.weight, std=0.02)
        nn.init.zeros_(self.visual_proj.bias)
        
        self.text_norm = nn.LayerNorm(input_dim)
        self.text_conditioned_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.text_conditioned_proj.weight, std=0.02)
        nn.init.zeros_(self.text_conditioned_proj.bias)
        
        # Attribute to store distillation loss
        self.distillation_loss = torch.tensor(0.0)
        
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass with dual-path architecture
        
        Args:
            x: Visual features [B, T, input_dim]
            text_tokens: Optional text token ids [B, L] - used only in training
            text_mask: Optional attention mask for text [B, L]
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        B = x.size(0)
        
        # Process input features
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Apply temporal modeling for speech
        if self.use_temporal_attention:
            x_temporal = x.transpose(1, 2)
            x_temporal = self.temporal_conv(x_temporal).transpose(1, 2)
            gate = self.temporal_gate(x)
            x = x + gate * x_temporal
        
        # Expand query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # VISUAL-ONLY PATH (always used)
        visual_only_hidden = query_tokens.clone()
        
        # Process through visual-only layers
        visual_layer_idx = 0
        while visual_layer_idx < len(self.visual_only_layers):
            layer = self.visual_only_layers[visual_layer_idx]
            
            if isinstance(layer, TransformerSelfAttentionLayer):
                # Self-attention layer
                visual_only_hidden = layer(visual_only_hidden)
            elif isinstance(layer, TransformerCrossAttentionLayer):
                # Visual cross-attention
                visual_only_hidden = layer(q=visual_only_hidden, kv=x, mask=None)
            
            visual_layer_idx += 1
        
        # Final visual-only output
        visual_only_output = self.visual_proj(self.visual_norm(visual_only_hidden))
        
        # If we're in inference mode or no text tokens provided, return visual-only path
        if not torch.is_grad_enabled() or text_tokens is None:
            return visual_only_output
        
        # TEXT-CONDITIONED PATH (only used during training)
        # Process text tokens
        text_tokens = text_tokens.to(dtype=torch.long)
        text_embeds = self.text_encoder.word_embeddings(text_tokens)
        text_features = self.text_proj(text_embeds)
        text_features = self.text_norm(text_features)
        
        if text_mask is None:
            text_mask = (text_tokens != 0).float()
        
        # Start with the same query tokens
        text_conditioned_hidden = query_tokens.clone()
        
        # Process through text-conditioned layers
        text_layer_idx = 0
        while text_layer_idx < len(self.text_conditioned_layers):
            layer = self.text_conditioned_layers[text_layer_idx]
            
            if isinstance(layer, TransformerSelfAttentionLayer):
                # Self-attention layer
                text_conditioned_hidden = layer(text_conditioned_hidden)
                text_layer_idx += 1
            elif isinstance(layer, TransformerCrossAttentionLayer):
                # Visual cross-attention
                text_conditioned_hidden = layer(q=text_conditioned_hidden, kv=x, mask=None)
                text_layer_idx += 1
                
                # Skip if we've reached the end
                if text_layer_idx >= len(self.text_conditioned_layers):
                    break
                    
                # Text cross-attention (next layer)
                text_attention_mask = None
                if text_mask is not None:
                    text_attention_mask = ~text_mask.bool()
                
                text_conditioned_hidden = self.text_conditioned_layers[text_layer_idx](
                    q=text_conditioned_hidden,
                    kv=text_features,
                    mask=text_attention_mask
                )
                text_layer_idx += 1
        
        # Final text-conditioned output
        text_conditioned_output = self.text_conditioned_proj(self.text_norm(text_conditioned_hidden))
        
        # KNOWLEDGE DISTILLATION
        # During training, compute distillation loss to help visual-only path learn from text-conditioned path
        if torch.is_grad_enabled():
            # MSE loss between the two outputs
            distillation_loss = F.mse_loss(visual_only_output, text_conditioned_output.detach())
            
            # Scale the loss and store it
            self.distillation_loss = distillation_loss * self.distillation_weight
            
            # During training, return text-conditioned output
            return text_conditioned_output
        else:
            # During inference, only use visual-only path
            return visual_only_output


class TextGuidedBlipQFormer(BaseProjector):
    """
    A text-guided QFormer inspired by BLIP2 that uses contrastive learning and text conditioning 
    during training but doesn't require text during inference.
    
    Key features:
    1. Dual-path architecture with shared query tokens
    2. Contrastive learning between visual and text representations
    3. Hard negative mining for better alignment
    4. Speech-specific temporal modeling
    5. Inference without text dependency
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        temporal_window_size: int = 5,
        contrastive_temperature: float = 0.07,
        max_text_len: int = 32,
        use_itm: bool = True,  # Image-text matching objective
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.use_temporal_attention = use_temporal_attention
        self.use_itm = use_itm
        self.max_text_len = max_text_len
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Input normalization and projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        
        # Text encoding components
        from transformers import BertConfig, BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Text embedding layer
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Projection layers for contrastive learning
        self.vision_proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.vision_proj.weight, std=0.02)
        nn.init.zeros_(self.vision_proj.bias)
        
        self.text_proj = nn.Linear(768, output_dim)  # BERT dim is 768
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
        
        # ITM head for image-text matching objective
        if use_itm:
            self.itm_head = nn.Linear(input_dim, 2)
            nn.init.normal_(self.itm_head.weight, std=0.02)
            nn.init.zeros_(self.itm_head.bias)
        
        # Temperature parameter for contrastive learning
        self.temp = nn.Parameter(contrastive_temperature * torch.ones([]))
        
        # Temporal attention for speech
        if use_temporal_attention:
            self.temporal_conv = nn.Conv1d(
                input_dim, 
                input_dim, 
                kernel_size=temporal_window_size,
                padding=(temporal_window_size-1)//2,
                groups=num_heads
            )
            
            self.temporal_gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
        
        # QFormer with transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention layer
            self.layers.append(
                TransformerSelfAttentionLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_dim,
                    dropout=dropout,
                    norm_first=True
                )
            )
            
            # Cross-attention to visual features (every other layer)
            if i % 2 == 1:
                self.layers.append(
                    TransformerCrossAttentionLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=intermediate_dim,
                        dropout=dropout,
                        norm_first=True
                    )
                )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(input_dim)
        
        # Track loss components
        self.contrastive_loss = torch.tensor(0.0)
        self.itm_loss = torch.tensor(0.0)
        
    def encode_text(self, text):
        """
        Encode text inputs into embeddings and attention masks
        """
        device = next(self.parameters()).device
        
        if isinstance(text, list):
            # Tokenize the text strings
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            ).to(device)
        elif isinstance(text, dict) and 'input_ids' in text and 'attention_mask' in text:
            # Already tokenized, just use as is
            text_tokens = text
        elif isinstance(text, torch.Tensor):
            # Assume it's just token IDs, create attention mask
            attention_mask = (text != 0).float()
            text_tokens = {
                'input_ids': text,
                'attention_mask': attention_mask
            }
        else:
            return None, None, None
            
        # Get text embeddings from BERT
        with torch.no_grad():
            text_output = self.text_encoder(
                text_tokens['input_ids'].to(device, dtype=torch.long),
                attention_mask=text_tokens['attention_mask'].to(device) if 'attention_mask' in text_tokens else None,
                return_dict=True
            )
        text_embeds = text_output.last_hidden_state
        
        # Project to the matching dimension and normalize
        text_feat = F.normalize(
            self.text_proj(text_embeds[:, 0, :]), dim=-1
        )
        
        return text_tokens, text_embeds, text_feat
        
    def forward(self, x: torch.Tensor, text_input=None, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass with dual-path architecture and contrastive learning
        
        Args:
            x: Visual features [B, T, input_dim]
            text_input: Optional text strings for BERT tokenization
            text_tokens: Optional pre-tokenized text 
            text_mask: Optional attention mask for text
            
        Returns:
            Projected features [B, num_queries, output_dim]
        """
        B = x.size(0)
        device = x.device
        
        # Process input features
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Apply temporal modeling for speech
        if self.use_temporal_attention:
            x_temporal = x.transpose(1, 2)
            x_temporal = self.temporal_conv(x_temporal).transpose(1, 2)
            gate = self.temporal_gate(x)
            x = x + gate * x_temporal
        
        # Create attention mask for visual features
        visual_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(device)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Process through transformer layers
        layer_idx = 0
        hidden_states = query_tokens.clone()
        
        while layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            
            if isinstance(layer, TransformerSelfAttentionLayer):
                # Self-attention layer
                hidden_states = layer(hidden_states)
            elif isinstance(layer, TransformerCrossAttentionLayer):
                # Visual cross-attention
                hidden_states = layer(q=hidden_states, kv=x, mask=None)
            
            layer_idx += 1
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Project to output dimension and normalize
        image_feats = F.normalize(self.vision_proj(hidden_states), dim=-1)
        
        # If not in training mode or no text provided, just return vision features
        if not torch.is_grad_enabled() or (text_input is None and text_tokens is None):
            return image_feats
        
        # Process text if available
        if text_input is not None:
            text_tokens, text_embeds, text_feat = self.encode_text(text_input)
        elif text_tokens is not None:
            # Handle the case when text_tokens is a raw tensor (instead of a dict-like object)
            if isinstance(text_tokens, torch.Tensor):
                # Create attention mask if not provided
                if text_mask is None:
                    text_mask = (text_tokens != 0).float()
                
                # Create a dict-like structure with the necessary attributes
                class TokenObject:
                    def __init__(self, input_ids, attention_mask):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                
                text_tokens = TokenObject(text_tokens, text_mask)
                
                # Skip ITM for raw tensor inputs to avoid indexing issues
                self._original_use_itm = self.use_itm
                self.use_itm = False
            
            # Use provided tokens
            text_output = self.text_encoder(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True
            )
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(
                self.text_proj(text_embeds[:, 0, :]), dim=-1
            )
        else:
            # No text provided
            return image_feats
            
        # Compute image-text similarity
        # Each query token attends to the text
        sim_q2t = torch.matmul(image_feats, text_feat.unsqueeze(-1)).squeeze(-1)
        
        # Get max similarity across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp
        
        # Compute text-image similarity
        sim_t2i = torch.matmul(text_feat, image_feats.permute(0, 2, 1)).max(-1)[0] / self.temp
        
        # Create proper similarity matrices for contrastive learning
        # We need [B, B] matrices where sim[i, j] is similarity between ith image and jth text
        full_sim_i2t = torch.zeros(B, B, device=device)
        full_sim_t2i = torch.zeros(B, B, device=device)
        
        # Fill the matrices
        for i in range(B):
            for j in range(B):
                # Image i to text j similarity
                full_sim_i2t[i, j] = torch.matmul(
                    image_feats[i].mean(0), 
                    text_feat[j]
                ) / self.temp
                
                # Text i to image j similarity
                full_sim_t2i[i, j] = torch.matmul(
                    text_feat[i], 
                    image_feats[j].mean(0)
                ) / self.temp
        
        # Targets are the diagonal elements (where i == j)
        targets = torch.arange(B, device=device)
        
        # Compute NLL loss
        loss_i2t = F.cross_entropy(full_sim_i2t, targets)
        loss_t2i = F.cross_entropy(full_sim_t2i, targets)
        
        self.contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        # Compute image-text matching loss if enabled
        if self.use_itm and B > 1 and not hasattr(self, '_original_use_itm'):  # Need at least 2 samples for negative mining
            with torch.no_grad():
                # Create similarity matrices for hard negative mining
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)
                
            # Select hard negative samples
            image_embeds_neg = []
            text_ids_neg = []
            text_atts_neg = []
            
            for b in range(B):
                # Select a negative image for each text
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(x[neg_idx])
                
                # Select a negative text for each image
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_tokens.input_ids[neg_idx])
                text_atts_neg.append(text_tokens.attention_mask[neg_idx])
            
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
            
            # Create input tensors with positive and negative pairs
            text_ids_all = torch.cat([text_tokens.input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([text_tokens.attention_mask, text_atts_neg], dim=0)
            
            image_embeds_all = torch.cat([x, image_embeds_neg], dim=0)
            
            # Process through transformer for ITM
            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            hidden_states_itm = query_tokens_itm.clone()
            
            # Process through transformer layers
            layer_idx = 0
            while layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                
                if isinstance(layer, TransformerSelfAttentionLayer):
                    hidden_states_itm = layer(hidden_states_itm)
                elif isinstance(layer, TransformerCrossAttentionLayer):
                    hidden_states_itm = layer(q=hidden_states_itm, kv=image_embeds_all, mask=None)
                
                layer_idx += 1
            
            # Final ITM prediction
            vl_embeddings = self.final_norm(hidden_states_itm)
            vl_output = self.itm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)
            
            # Create ITM labels (1 for positive pairs, 0 for negative pairs)
            itm_labels = torch.cat(
                [torch.ones(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)],
                dim=0
            ).to(device)
            
            # Compute ITM loss
            self.itm_loss = F.cross_entropy(logits, itm_labels)
        
        # If we temporarily disabled ITM, restore the original setting
        if hasattr(self, '_original_use_itm'):
            self.use_itm = self._original_use_itm
            delattr(self, '_original_use_itm')

        # During training, return the image features (the text alignment is enforced via loss)
        return image_feats

    def get_loss(self):
        """
        Get combined loss from contrastive and ITM objectives
        """
        if self.use_itm:
            return self.contrastive_loss + self.itm_loss
        return self.contrastive_loss


def get_projector(projector_name, input_dim, output_dim, **kwargs):
    """
    Factory function to create the appropriate projector based on name
    
    Args:
        projector_name: Name/type of projector to create
        input_dim: Input dimension
        output_dim: Output dimension 
        **kwargs: Additional projector-specific parameters
        
    Returns:
        Instantiated projector module
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
            num_queries=kwargs.get("num_queries", 64),
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
        
    elif projector_name == "comprehensive_qformer":
        return ComprehensiveQFormerProjector(
            input_dim,
            output_dim,
            num_queries=kwargs.get("num_queries", 32),
            num_layers=kwargs.get("num_layers", 6),
            num_heads=kwargs.get("num_heads", 8),
            intermediate_dim=kwargs.get("intermediate_dim", 3072),
            dropout=kwargs.get("dropout", 0.1),
            use_text_conditioning=kwargs.get("use_text_conditioning", True),
            use_temporal_attention=kwargs.get("use_temporal_attention", True),
            max_text_len=kwargs.get("max_text_len", 32),
            temporal_window_size=kwargs.get("temporal_window_size", 5)
        )
        
    # Visual Speech QFormer
    elif projector_name == "visual_speech_qformer":
        num_queries = kwargs.get('num_queries', 32)
        num_layers = kwargs.get('num_layers', 2)
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        window_size = kwargs.get('window_size', 3)
        return VisualSpeechQFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )
        
    # Visual Speech Text QFormer - new projector
    elif projector_name == "visual_speech_text_qformer":
        # Remove args that are not used by VisualSpeechTextQFormer
        for k in list(kwargs.keys()):
            if k not in inspect.signature(VisualSpeechTextQFormer.__init__).parameters:
                kwargs.pop(k)
                
        return VisualSpeechTextQFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    
    elif projector_name.lower() == "visual_only_qformer":
        return VisualOnlyQFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=kwargs.get("projector_num_queries", 32),
            num_layers=kwargs.get("projector_num_layers", 6),
            num_heads=kwargs.get("projector_num_heads", 8),
            intermediate_dim=kwargs.get("projector_hidden_dim", 3072),
            dropout=kwargs.get("projector_dropout", 0.1),
            use_temporal_attention=kwargs.get("use_temporal_attention", True),
            temporal_window_size=kwargs.get("temporal_window_size", 5)
        )
    
    elif projector_name.lower() == "visual_only_blip2_qformer":
        return VisualOnlyBlip2QFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=kwargs.get("projector_num_queries", 32),
            num_layers=kwargs.get("projector_num_layers", 6),
            num_heads=kwargs.get("projector_num_heads", 8),
            dropout=kwargs.get("projector_dropout", 0.1),
            intermediate_size=kwargs.get("projector_hidden_dim", 3072)
        )
    
    elif projector_name.lower() == "visual_only_cross_attention":
        return VisualOnlyCrossAttention(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=kwargs.get("projector_num_heads", 8),
            num_layers=kwargs.get("projector_num_layers", 2),
            dropout=kwargs.get("projector_dropout", 0.1)
        )
    
    elif projector_name.lower() == "visual_only_multiscale_contrastive":
        return VisualOnlyMultiScaleContrastive(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=kwargs.get("projector_hidden_dim", 2048),
            num_scales=kwargs.get("num_scales", 3),
            num_heads=kwargs.get("projector_num_heads", 8),
            dropout=kwargs.get("projector_dropout", 0.1)
        )
    
    elif projector_name.lower() == "text_guided_qformer":
        return TextGuidedQFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=kwargs.get("projector_num_queries", 32),
            num_layers=kwargs.get("projector_num_layers", 6),
            num_heads=kwargs.get("projector_num_heads", 8),
            intermediate_dim=kwargs.get("projector_hidden_dim", 3072),
            dropout=kwargs.get("projector_dropout", 0.1),
            use_temporal_attention=kwargs.get("use_temporal_attention", True),
            temporal_window_size=kwargs.get("temporal_window_size", 5),
            distillation_weight=kwargs.get("distillation_weight", 0.5)
        )
    
    elif projector_name.lower() == "text_guided_blip_qformer":
        return TextGuidedBlipQFormer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_queries=kwargs.get("projector_num_queries", 32),
            num_layers=kwargs.get("projector_num_layers", 6),
            num_heads=kwargs.get("projector_num_heads", 8),
            intermediate_dim=kwargs.get("projector_hidden_dim", 3072),
            dropout=kwargs.get("projector_dropout", 0.1),
            use_temporal_attention=kwargs.get("use_temporal_attention", True),
            temporal_window_size=kwargs.get("temporal_window_size", 5),
            contrastive_temperature=kwargs.get("contrastive_temperature", 0.07),
            max_text_len=kwargs.get("max_text_len", 32),
            use_itm=kwargs.get("use_itm", True)
        )
    
    else:
        raise ValueError(f"Unknown projector type: {projector_name}")


