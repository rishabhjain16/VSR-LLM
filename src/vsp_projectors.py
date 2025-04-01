# Copyright (c) 2023
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
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
        
        # Check text input and create mask if needed
        if self.use_text_conditioning and text_tokens is not None:
            if not isinstance(text_tokens, torch.Tensor):
                logger.warning(f"text_tokens is not a tensor, but {type(text_tokens)}. Ignoring text input.")
                text_tokens = None
                text_mask = None
            elif text_mask is None:
                text_mask = (text_tokens != 0).float()
                
        # Process text if available
        text_features = None
        if self.use_text_conditioning and text_tokens is not None:
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
                if (self.use_text_conditioning and text_features is not None and 
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
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, q, kv, mask=None):
        """
        Args:
            q: Query tensor [B, Q, D]
            kv: Key/value tensor [B, T, D]
            mask: Optional mask for kv
        
        Returns:
            Updated query tensor
        """
        # Apply layer norm first if using pre-norm
        if self.norm_first:
            x = q + self.dropout1(self.cross_attn(
                self.norm1(q), self.norm1(kv), self.norm1(kv), 
                key_padding_mask=mask)[0])
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(
                self.linear1(self.norm2(x))))))
        else:
            x = self.norm1(q + self.dropout1(self.cross_attn(
                q, kv, kv, key_padding_mask=mask)[0]))
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
            
            # Apply visual frontend for lip movement features
            x_conv = x.transpose(1, 2)  # [B, D, T]
            x_conv = self.visual_frontend(x_conv)
            x = x_conv.transpose(1, 2)  # [B, T, D]
            
            # Add subtle positional information to preserve temporal order in lip movements
            # Cast to float32 for better precision during position calculation
            orig_dtype = x.dtype
            position = torch.arange(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            position = position.expand(batch_size, -1, x.size(-1))
            position = position.to(torch.float32) / float(x.size(1))
            
            # Safer addition with position encoding - convert back to original dtype
            position = (0.1 * position).to(orig_dtype)
            x = x + position
            
            # Expand query tokens to batch size
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            
            # Concatenate query tokens with visual features
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
            
            # Process through transformer
            hidden_states = self.transformer(x_with_query, mask=mask)
            
            # Extract only the query outputs
            query_output = hidden_states[:, :query_len]
            
            # Final norm and projection
            query_output = self.norm(query_output)
            output = self.output_proj(query_output)
            
            return output
            
        except RuntimeError as e:
            # Handle mixed precision error explicitly
            if "cuda runtime error" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
                self.logger.warning(f"Mixed precision error in VisualSpeechQFormer: {e}")
                # Fall back to float32 computation
                with torch.cuda.amp.autocast(enabled=False):
                    return self.forward(x.float(), 
                                       None if text_tokens is None else text_tokens.float(), 
                                       text_mask)
            else:
                raise e


class VisualSpeechTextQFormer(VisualSpeechQFormer):
    """QFormer for visual speech processing with text conditioning for disambiguation
    
    This projector adds text conditioning to the VisualSpeechQFormer, allowing it to
    better disambiguate visually similar phonemes using text context. It is designed
    to be numerically stable with mixed precision training.
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
        super().__init__(
            input_dim=input_dim, 
            output_dim=output_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )
        
        # Add text processing
        self.text_dim = text_dim
        
        # Linear projection for text if dimensions don't match
        if text_dim != input_dim:
            self.text_projection = nn.Linear(text_dim, input_dim)
        else:
            self.text_projection = nn.Identity()
            
        # Cross-attention layers for text conditioning
        # Create a separate visual-text cross attention that's numerically stable
        self.cross_attn_layers = nn.ModuleList([
            StableCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads, 
                dropout=dropout
            ) for _ in range(text_num_layers)
        ])
    
    def forward(self, x: torch.Tensor, text_tokens=None, text_mask=None) -> torch.Tensor:
        """
        Forward pass for VisualSpeechTextQFormer, which enhances visual query features with text conditioning.
        
        Unlike the base VisualSpeechQFormer which ignores text input, this implementation actively uses 
        the text tokens to disambiguate visually similar phonemes via cross-attention.
        
        Args:
            x: Visual/audio features [B, T, input_dim]
            text_tokens: Text token ids [B, L, D] - actively used for cross-modal alignment
            text_mask: Attention mask for text (True for valid tokens, False for padding)
            
        Returns:
            Text-enhanced projected features [B, num_queries, output_dim]
        """
        try:
            # First get visual query outputs using parent class implementation
            # We override parent class forward but call it directly to avoid recursive loop
            with torch.no_grad():
                query_output = super().forward(x, None, None)
            
            # If no text conditioning provided, just return the visual query outputs
            if text_tokens is None:
                return query_output
                
            # Process text tokens through projection if needed
            text_features = self.text_projection(text_tokens)
            
            # Apply cross-attention between queries and text
            for layer in self.cross_attn_layers:
                # More stable cross-attention with better precision handling
                text_attention_mask = None if text_mask is None else ~text_mask.bool()
                query_output = layer(
                    query_output, 
                    text_features,
                    key_padding_mask=text_attention_mask
                )
                
            return query_output
            
        except RuntimeError as e:
            # Handle mixed precision error explicitly
            if "cuda runtime error" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
                self.logger.warning(f"Mixed precision error in VisualSpeechTextQFormer: {e}")
                # Fall back to float32 computation
                with torch.cuda.amp.autocast(enabled=False):
                    return self.forward(x.float(), 
                                       None if text_tokens is None else text_tokens.float(), 
                                       text_mask)
            else:
                raise e


class StableCrossAttentionLayer(nn.Module):
    """Cross-attention layer with improved numerical stability for mixed precision training"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN components
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation with better numerical stability
        self.activation = F.gelu
        
    def forward(self, x, kv, key_padding_mask=None):
        """Forward pass with improved numerical stability
        
        Args:
            x: Query tensor [B, seq_len, dim]
            kv: Key/value tensor [B, kv_len, dim]
            key_padding_mask: Mask for kv, True indicates invalid positions
            
        Returns:
            Tensor [B, seq_len, dim]
        """
        # Store original dtype
        orig_dtype = x.dtype
        
        # Pre-LayerNorm for better numerical stability
        x_norm = self.norm1(x)
        kv_norm = self.norm1(kv)
        
        # Cross-attention
        attn_output, _ = self.multihead_attn(
            query=x_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Residual connection with dropout
        x = x + self.dropout1(attn_output)
        
        # FFN with pre-LayerNorm
        x_norm = self.norm2(x)
        
        # FFN with residual
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_output)
        
        return x


def get_projector(projector_name, input_dim, output_dim, **kwargs):
    """
    Factory function to create a projector by name
    
    All projectors except for Linear and MLP support text-based conditioning
    via transcript tokens. This allows for direct visual-text alignment
    for better representation learning.
    
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
    
    else:
        raise ValueError(f"Unknown projector type: {projector_name}")


