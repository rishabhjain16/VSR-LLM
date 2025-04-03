#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys, logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from einops import repeat
import datetime
import math
import random
import time
import contextlib
import inspect

from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from omegaconf import II, MISSING

# Import our new module for LoRA target module selection
from .vsp_lora import get_target_modules, LORA_CONFIG

# Import the projector module
from .vsp_projectors import get_projector

# Get logger
logger = logging.getLogger(__name__)

MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class VSPLLMConfig(FairseqDataclass):
    w2v_path: str = field(
        default=os.path.join("checkpoints", "large_vox_iter5.pt"), 
        metadata={"help": "path to hubert model"}
    )
    llm_ckpt_path: str = field(
        default=MISSING, metadata={"help": "path to llama model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
                    "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    masking_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    decoder_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension"}
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )

    # Add new fields for projector configuration
    projector_type: str = field(
        default="visual_speech_qformer",
        metadata={"help": "Type of projector to use (linear, mlp, qformer, cross_attention, blip2_qformer, comprehensive_qformer, visual_speech_qformer, visual_speech_text_qformer, multiscale_contrastive)"}
    )
    projector_hidden_dim: Optional[int] = field(
        default=None, 
        metadata={"help": "Hidden dimension for projector (if applicable)"}
    )
    projector_num_layers: int = field(
        default=2, 
        metadata={"help": "Number of layers in projector (if applicable)"}
    )
    projector_num_heads: int = field(
        default=8, 
        metadata={"help": "Number of attention heads in projector (if applicable)"}
    )
    projector_dropout: float = field(
        default=0.1, 
        metadata={"help": "Dropout rate in projector (if applicable)"}
    )
    projector_num_queries: int = field(
        default=32, 
        metadata={"help": "Number of queries in query-based projectors (if applicable)"}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to data directory. DEPRECATED!"},
    )

    # Add new fields for CTC configuration
    use_ctc: bool = field(
        default=False,
        metadata={"help": "Whether to use CTC auxiliary loss during training"}
    )
    ctc_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for CTC loss (0.3 means 30% CTC, 70% LM)"}
    )
    ctc_weight_decode: float = field(
        default=0.3,
        metadata={"help": "Weight for CTC during inference (0.3 means 30% CTC, 70% LM)"}
    )
    ctc_blank_idx: Optional[int] = field(
        default=0, 
        metadata={"help": "Index to use for blank token in CTC (defaults to 0)"}
    )
    ctc_feature_source: str = field(
        default="projector",
        metadata={"help": "Source of features for CTC loss: 'encoder' (raw AV-HuBERT) or 'projector' (after projection)"}
    )
    # Add a field for CTC vocabulary size (always using phonetic vocabulary)
    ctc_vocab_size: int = field(
        default=50,
        metadata={"help": "Size of the phonetic vocabulary for CTC"}
    )



class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward_(self, source, padding_mask, **kwargs):
        src ={}
        src['video'] = source
        src['audio'] = None
        w2v_args = {
            "source": src,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def forward(self, source, padding_mask, **kwargs):
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
            }

            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)


            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }
 

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out




@register_model("vsp_llm", dataclass=VSPLLMConfig)
class avhubert_llm_seq2seq_cluster_count(BaseFairseqModel):
    def __init__(self, encoder, decoder, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        
        # Initialize num_updates for logging
        self.num_updates = 0
        self.batch_counter = 0  # Add batch counter for more frequent logging
        
        # Get hidden size dynamically from the decoder's configuration
        if hasattr(decoder, 'config') and hasattr(decoder.config, 'hidden_size'):
            hidden_size = decoder.config.hidden_size
        else:
            # Fallback to default for Llama models
            hidden_size = 4096
            
        # Import and initialize the projector
        projector_kwargs = {
            'input_dim': 1024,
            'output_dim': hidden_size,
        }
        
        # Add optional arguments if they're set in config
        if cfg.projector_hidden_dim is not None:
            projector_kwargs['hidden_dim'] = cfg.projector_hidden_dim
        if cfg.projector_num_layers > 0:
            projector_kwargs['num_layers'] = cfg.projector_num_layers
        if cfg.projector_num_heads > 0:
            projector_kwargs['num_heads'] = cfg.projector_num_heads
        if cfg.projector_dropout > 0:
            projector_kwargs['dropout'] = cfg.projector_dropout
        if cfg.projector_num_queries > 0:
            projector_kwargs['num_queries'] = cfg.projector_num_queries
            
        # Create the projector
        self.avfeat_to_llm = get_projector(cfg.projector_type, **projector_kwargs)
        
        # Log the projector type being used
        logger.info(f"==========================================================")
        logger.info(f"INITIALIZING MODEL WITH PROJECTOR TYPE: {cfg.projector_type}")
        logger.info(f"==========================================================")
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        
        # Track whether we've logged the projector shape
        self.logged_projector_shape = False
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            # Try to create a flexible path that works across different environments
            w2v_paths_to_try = [
                cfg.w2v_path,  # Original path (could be absolute or relative)
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.w2v_path),  # Try relative to project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "large_vox_iter5.pt"),  # Try in project checkpoints dir
                os.path.abspath(cfg.w2v_path)  # Try absolute path
            ]
            
            # Try each path in order
            w2v_path = None
            for path in w2v_paths_to_try:
                if os.path.exists(path):
                    w2v_path = path
                    logger.info(f"Found AV-HuBERT model at: {w2v_path}")
                    break
            
            if w2v_path is None:
                raise FileNotFoundError(
                    f"Could not find AV-HuBERT model. Tried the following paths: {w2v_paths_to_try}. "
                    f"Please ensure the model file exists or specify the correct path."
                )
                
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )
        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        decoder_4bit = AutoModelForCausalLM.from_pretrained(cfg.llm_ckpt_path, quantization_config=bnb_config)            

        # Use our new function to get target modules based on model architecture
        if hasattr(decoder_4bit.config, 'model_type'):
            model_type = decoder_4bit.config.model_type.lower()
            print(f"\n=== Training LoRA on model type: {model_type} ===")
        else:
            model_type = "unknown"
            print("\n=== Training LoRA on unknown model type ===")
        
        # Get target modules using our new helper function
        target_modules = get_target_modules(decoder_4bit, verbose=True)
        print(f"=== LoRA target modules: {target_modules} ===\n")
        
        # Use fixed LoRA parameters from vsp_lora.py
        from .vsp_lora import LORA_CONFIG
        
        config = LoraConfig(
            r=LORA_CONFIG['r'], 
            lora_alpha=LORA_CONFIG['lora_alpha'], 
            target_modules=target_modules, 
            lora_dropout=LORA_CONFIG['lora_dropout'], 
            bias=LORA_CONFIG['bias'], 
            task_type=LORA_CONFIG['task_type'] 
        )

        decoder_4bit = get_peft_model(decoder_4bit, config)
        decoder_4bit.print_trainable_parameters()
        
        return avhubert_llm_seq2seq_cluster_count(encoder, decoder_4bit, cfg)
    
    def forward(self, **kwargs):
        # Increment batch counter for logging
        self.batch_counter += 1
        
        # Always update CTC status at the start of forward to ensure it's correct
        # This prevents issues when transitioning between train and validation
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'use_ctc'):
            self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # First get encoder output
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
    
        # Get the original size before applying projector
        orig_seq_len = output['encoder_out'].size(1)
        
        # Get transcript information for cross-modal alignment
        labels = kwargs['target_list'].clone()
        
        # Get labels embedding directly from the model
        if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
            labels_embedding = self.decoder.model.model.embed_tokens(labels)
        elif hasattr(self.decoder, 'model'):
            labels_embedding = self.decoder.model.embed_tokens(labels)
        else:
            labels_embedding = self.decoder.embed_tokens(labels)
        
        # Simplified rule for text-based projectors:
        # Linear and MLP are cluster-based, everything else is text-based
        is_text_based = self.cfg.projector_type.lower() not in ['linear', 'mlp']
        
        # Apply projector based on type
        alignment_loss = torch.tensor(0.0, device=output['encoder_out'].device)
        
        if is_text_based:
            # For all text-based projectors, always pass transcript tokens embeddings
            if not self.logged_projector_shape:
                logger.info(f"Using transcript tokens for vision-text mapping in {self.cfg.projector_type}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (labels != 0)
            
            # Convert embeddings to match encoder output dtype
            labels_embedding_matched = labels_embedding.to(dtype=output['encoder_out'].dtype)
            
            # Check projector's forward method signature to determine which parameters to pass
            forward_params = inspect.signature(self.avfeat_to_llm.forward).parameters
            
            # Base parameters that all projectors accept
            projector_args = {
                "x": output['encoder_out']
            }
            
            # Add text_tokens parameter if it exists in the signature
            if "text_tokens" in forward_params:
                projector_args["text_tokens"] = labels_embedding_matched
                
            # Add text_mask parameter if it exists in the signature
            if "text_mask" in forward_params:
                projector_args["text_mask"] = text_mask
                
            # For QFormer-specific implementations that use text_embeddings instead of text_tokens
            if "text_embeddings" in forward_params and "text_tokens" not in forward_params:
                projector_args["text_embeddings"] = labels_embedding_matched
                
            # Pass the appropriate parameters to the projector
            output['encoder_out'] = self.avfeat_to_llm(**projector_args)
        else:
            # For non-text-based projectors (linear, mlp), just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        
        # Check if we're using a query-based projector that changes sequence length
        proj_seq_len = output['encoder_out'].size(1)
        
        # Query-based projectors are the same as text-based in our case, 
        # since only linear and mlp are not query-based
        is_query_based = is_text_based
            
        # Handle different projector types
        if is_query_based:
            # For query-based projectors that return fixed number of tokens
            # We skip the cluster-based aggregation as these projectors already aggregate
            reduced_enc_out = output['encoder_out']
            # Only log the shape once
            if not self.logged_projector_shape:
                logger.info(f"Using query-based projector output with shape: {reduced_enc_out.size()}")
                self.logged_projector_shape = True
        else:
            # For projectors that maintain sequence length, use cluster-based aggregation
            cluster_counts = kwargs['source']['cluster_counts'][0]  # tensor list
            
            results_tensor = []
            start_idx = 0
            for clutser_num in cluster_counts:
                end_idx = start_idx + clutser_num
                slice = output['encoder_out'][:,start_idx:end_idx,:]
                
                # Original method: simple mean aggregation
                mean_avg = slice.mean(dim=1, keepdim=True)
                results_tensor.append(mean_avg)
                
                start_idx = end_idx
    
            # Verify we processed the entire sequence
            assert(cluster_counts.sum().item() == proj_seq_len)
            reduced_enc_out = torch.cat(results_tensor, dim=1)
            # Only log the shape once
            if not self.logged_projector_shape:
                logger.info(f"Using mean cluster aggregation with output shape: {reduced_enc_out.size()}")
                self.logged_projector_shape = True
        
        B, T, D = reduced_enc_out.size()
        instruction = kwargs['source']['text']
        
        # Get instruction embedding with model structure awareness
        try:
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                instruction_embedding = self.decoder.model.model.embed_tokens(instruction)
            elif hasattr(self.decoder, 'model'):
                instruction_embedding = self.decoder.model.embed_tokens(instruction)
            else:
                instruction_embedding = self.decoder.embed_tokens(instruction)
        except AttributeError:
            # Fallback to get_input_embeddings method
            embedding_layer = self.decoder.get_input_embeddings()
            instruction_embedding = embedding_layer(instruction)

        llm_input = torch.cat((instruction_embedding, reduced_enc_out, labels_embedding), dim=1)
        
        # Prepare labels for loss calculation (used by all model types)
        llm_labels = labels.clone()
        llm_labels[llm_labels == 0] = -100
        
        _, instruction_embedding_t, _ = instruction_embedding.size()
        target_ids = torch.full((B, T + instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)
        
        # If CTC is enabled and we're in training mode
        if self._use_ctc_in_forward:
            # Get CTC features based on the configured source
            if self.cfg.ctc_feature_source == "encoder":
                # Use raw encoder output for CTC
                ctc_features = reduced_enc_out.clone()
                
                # Only calculate logits if using encoder features
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
                
                # Initialize input_lengths to None to avoid UnboundLocalError
                input_lengths = None
                
                # For encoder output, be careful with input lengths
                if hasattr(self, 'encoder_outputs_per_layer'):
                    # Use encoder-specific information if available
                    input_lengths = self.encoder_outputs_per_layer.get('lengths', None)
                
                if input_lengths is None:
                    # Fallback to assuming all time steps are valid
                    input_lengths = torch.full((reduced_enc_out.size(0),), 
                                            reduced_enc_out.size(1),
                                            device=reduced_enc_out.device,
                                            dtype=torch.long)
                
                # Log which feature source is being used
                if self.batch_counter % 20 == 0:
                    logger.info(f"Using raw encoder output for CTC, shape: {ctc_features.shape}")
            else:
                # Use projector output for CTC (default)
                ctc_features = reduced_enc_out.clone()
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
                
                # For projector output, be more careful with input lengths
                # Key difference: projector outputs might have different sequence length characteristics
                # compared to encoder outputs, especially after clustering/aggregation
                if is_query_based:
                    # For query-based projectors, each output token should be considered
                    input_lengths = torch.full((ctc_features.size(0),), 
                                          ctc_features.size(1),
                                          device=ctc_features.device,
                                          dtype=torch.long)
                else:
                    # For clustering-based approaches, the sequence reflects the number of clusters
                    # which should match the number of distinct segments in the input
                    input_lengths = torch.full((ctc_features.size(0),), 
                                          ctc_features.size(1),
                                          device=ctc_features.device,
                                          dtype=torch.long)
                
                # Log which feature source is being used
                if self.batch_counter % 20 == 0:
                    logger.info(f"Using projector output for CTC, shape: {ctc_features.shape}")
                
            # Apply log softmax to CTC logits
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
            
            # Map target to phonetic indices
            # Create mapping tensor for LLM token IDs to phonetic IDs (map everything to <unk>)
            # In a real implementation, you would have a more sophisticated mapping
            # For now, we'll just use a simple strategy for testing
            unk_idx = self.phonetic_vocab.get('<unk>', 1)  # Default to 1 if not found
            
            # Clone target to avoid modifying the original
            phonetic_target = torch.full_like(labels, unk_idx)
            
            # Get non-padding tokens
            non_pad_mask = (labels != 0)
            
            # Set non-pad tokens to unk_idx (this is a simplified approach)
            # In a real implementation, you would map each token to an appropriate phonetic sequence
            phonetic_target = phonetic_target * non_pad_mask
            
            # Use phonetic_target for CTC loss calculation
            target = phonetic_target
            
            # Remove padding tokens for CTC loss
            pad_mask = (target != 0)  # 0 is padding token
            target_lengths = pad_mask.sum(-1)
            
            # Add diagnostic logs to help debug CTC issues
            if self.batch_counter % 20 == 0:
                non_zero_targets = (target_lengths > 0).sum().item()
                total_targets = target_lengths.size(0)
                logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                logger.info(f"CTC input_lengths: min={input_lengths.min().item()}, max={input_lengths.max().item()}")
                logger.info(f"CTC target_lengths: min={target_lengths.min().item()}, max={target_lengths.max().item()}")
                
                # Check if we have invalid CTC constraints (input shorter than target)
                invalid_lens = (input_lengths < target_lengths).sum().item()
                if invalid_lens > 0:
                    logger.warning(f"CTC constraint violation: {invalid_lens} sequences have input_length < target_length")
                
                # Log vocabulary information
                logger.info(f"Using phonetic vocabulary for CTC targets, size: {self.ctc_vocab_size}")
            
            # Calculate CTC loss
            try:
                # Check for empty targets or input
                if (target_lengths == 0).all() or (input_lengths == 0).all():
                    logger.warning(f"Empty CTC targets or inputs detected - CTC loss will be ignored")
                    ctc_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
                else:
                    # CTC needs float32 for loss calculation
                    log_probs_float = log_probs.float()
                    ctc_loss = F.ctc_loss(
                        log_probs_float,
                        target,
                        input_lengths,
                        target_lengths,
                        blank=self.cfg.ctc_blank_idx,
                        reduction="mean",
                        zero_infinity=True,
                    )
                    
                    # Check if loss is suspiciously close to zero
                    if ctc_loss.item() < 1e-6:
                        logger.warning(f"CTC loss is suspiciously close to zero: {ctc_loss.item()}")
                        
                        # Check logits for possible numerical issues
                        max_logit = ctc_logits.max().item()
                        min_logit = ctc_logits.min().item()
                        mean_logit = ctc_logits.mean().item()
                        logger.warning(f"CTC logits stats: min={min_logit:.4f}, max={max_logit:.4f}, mean={mean_logit:.4f}")
                
                # Combine losses with weighting
                # Convert to same dtype for combination
                # Add alignment loss if available (for cross-modal learning)
                if is_text_based and alignment_loss.item() > 0:
                    # Add alignment loss with a weight (e.g., 0.1)
                    alignment_weight = 0.1
                    total_loss = (1 - self.cfg.ctc_weight - alignment_weight) * lm_loss + \
                                self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype) + \
                                alignment_weight * alignment_loss.to(lm_loss.dtype)
                    
                    # Log all losses
                    if self.batch_counter % 20 == 0:
                        lm_loss_val = float(lm_loss.detach().cpu().item())
                        ctc_loss_val = float(ctc_loss.detach().cpu().item())
                        align_loss_val = float(alignment_loss.detach().cpu().item())
                        logger.info(f"Batch {self.batch_counter}: LM loss: {lm_loss_val:.4f}, "
                                   f"CTC loss: {ctc_loss_val:.4f}, "
                                   f"Alignment loss: {align_loss_val:.4f}")
                else:
                    # Standard weighting without alignment loss
                    total_loss = (1 - self.cfg.ctc_weight) * lm_loss + self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype)
                    
                    # Log losses occasionally
                    if self.batch_counter % 20 == 0:
                        lm_loss_val = float(lm_loss.detach().cpu().item())
                        ctc_loss_val = float(ctc_loss.detach().cpu().item())
                        logger.info(f"Batch {self.batch_counter}: LM loss: {lm_loss_val:.4f}, CTC loss: {ctc_loss_val:.4f}")
                
            except Exception as e:
                logger.warning(f"CTC loss calculation failed: {e}")
                # Add detailed exception information for debugging
                import traceback
                logger.warning(f"CTC exception traceback: {traceback.format_exc()}")
                logger.warning(f"CTC input shape: {ctc_features.shape}, target shape: {target.shape}")
                logger.warning(f"input_lengths: {input_lengths}")
                logger.warning(f"target_lengths: {target_lengths}")
                total_loss = lm_loss  # Fallback to just LM loss
        
        # Return combined loss, logits, and labels
        return total_loss, llm_out.logits, llm_labels


    @torch.no_grad()
    def generate(self,
                num_beams=20,
                max_length=30,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=0.0,
                use_ctc_decoding=True,  # Changed default to True
                ctc_beam_size=10,
                ctc_weight_decode=0.3,  # Set default weight for hybrid decoding
                return_ctc_outputs=True,  # Added to return CTC outputs
                return_both_outputs=False,  # Return both LLM-only and hybrid outputs
                **kwargs,
                ):
        output = self.encoder(**kwargs)
        
        # Check if the projector supports text conditioning for alignment loss
        is_text_aware = any(qp in self.cfg.projector_type.lower() for qp in [
            "qformer", "blip2_qformer", "comprehensive_qformer", 
            "visual_speech_qformer", "visual_speech_text_qformer"
        ])
        
        # For all projectors, try to use text information if available
        if is_text_aware:
            # Get text tokens from kwargs (only instruction tokens are available during inference)
            text_tokens = kwargs['source']['text']
            
            # Add debug info about text tokens
            logger.info(f"Generate - Text tokens shape: {text_tokens.shape}, non-zero tokens: {(text_tokens != 0).sum().item()}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (text_tokens != 0)
            logger.info(f"Using instruction tokens in generate() mode (transcript tokens not available during inference)")
            
            # Get embeddings for the tokens
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                text_embeddings = self.decoder.model.model.embed_tokens(text_tokens)
            elif hasattr(self.decoder, 'model'):
                text_embeddings = self.decoder.model.embed_tokens(text_tokens)
            else:
                text_embeddings = self.decoder.embed_tokens(text_tokens)
            
            # Match dtype with encoder output for compatibility
            text_embeddings_matched = text_embeddings.to(dtype=output['encoder_out'].dtype)
            
            # Check projector's forward method signature to determine which parameters to pass
            forward_params = inspect.signature(self.avfeat_to_llm.forward).parameters
            
            # Base parameters that all projectors accept
            projector_args = {
                "x": output['encoder_out']
            }
            
            # Add text_tokens parameter if it exists in the signature
            if "text_tokens" in forward_params:
                projector_args["text_tokens"] = text_embeddings_matched
                
            # Add text_mask parameter if it exists in the signature
            if "text_mask" in forward_params:
                projector_args["text_mask"] = text_mask
                
            # For QFormer-specific implementations that use text_embeddings instead of text_tokens
            if "text_embeddings" in forward_params and "text_tokens" not in forward_params:
                projector_args["text_embeddings"] = text_embeddings_matched
                
            # Pass the appropriate parameters to the projector
            output['encoder_out'] = self.avfeat_to_llm(**projector_args)
        else:
            # For non-text-aware projectors, just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
            
        # Handle different projector types
        proj_seq_len = output['encoder_out'].size(1)
        orig_seq_len = kwargs['source']['video'].size(1)  # Original sequence length
        
        # First check projector type name (more reliable)
        is_query_based = any(qp in self.cfg.projector_type.lower() for qp in [
            "qformer", "visual_speech_qformer", "cross_attention",
            "blip2_qformer", "comprehensive_qformer",
            "visual_speech_text_qformer", "multiscale_contrastive"
        ])
        
        # Fallback to sequence length check if needed
        if not is_query_based:
            is_query_based = proj_seq_len != orig_seq_len
        
        if is_query_based:
            # For query-based projectors, use the output directly
            reduced_enc_out = output['encoder_out']
        else:
            # For projectors that maintain sequence length, use cluster-based aggregation
            cluster_counts = kwargs['source']['cluster_counts'][0]  # tensor list
            
            results_tensor = []
            start_idx = 0
            for clutser_num in cluster_counts:
                end_idx = start_idx + clutser_num
                slice = output['encoder_out'][:,start_idx:end_idx,:]
                
                # Original method: simple mean aggregation
                mean_avg = slice.mean(dim=1, keepdim=True)
                results_tensor.append(mean_avg)
                
                start_idx = end_idx
    
            # Verify we processed the entire sequence
            assert(cluster_counts.sum().item() == proj_seq_len)
            reduced_enc_out = torch.cat(results_tensor, dim=1)
            # Only log the shape once
            if not self.logged_projector_shape:
                logger.info(f"Using mean cluster aggregation with output shape: {reduced_enc_out.size()}")
                self.logged_projector_shape = True
        
        # Option to use CTC decoding
        if use_ctc_decoding and hasattr(self, 'ctc_vocab_size'):
            # Get CTC logits based on configured CTC head
            if self.cfg.ctc_feature_source == "encoder" and self.ctc_head_encoder is not None:
                ctc_features = output['encoder_out'].clone()
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
            elif self.ctc_head_projector is not None:
                ctc_features = reduced_enc_out.clone()
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
            else:
                # Fall back to normal generation if CTC head is not available
                logger.warning("CTC decoding requested but appropriate CTC head is not available")
                use_ctc_decoding = False
            
            if use_ctc_decoding:
                # Apply log softmax to CTC logits
                log_probs = F.log_softmax(ctc_logits, dim=-1)  # [B, T, V]
                
                # Simplified greedy decoding (we could implement beam search in a more sophisticated system)
                # This just demonstrates the basic approach
                best_paths = torch.argmax(log_probs, dim=-1)  # [B, T]
                
                # Remove repeated tokens and blanks (basic CTC decoding)
                decoded_seqs = []
                for seq in best_paths:
                    decoded = []
                    prev = -1  # Different from all valid indices
                    for token_idx in seq:
                        token = token_idx.item()
                        if token != prev and token != self.cfg.ctc_blank_idx:
                            decoded.append(token)
                        prev = token
                    
                    # Convert to tensor
                    decoded_tensor = torch.tensor(decoded, device=best_paths.device)
                    decoded_seqs.append(decoded_tensor)
                
                # If we're using a phonetic vocabulary, convert to a readable format
                if hasattr(self, 'idx_to_phonetic'):
                    readable_seqs = []
                    for seq in decoded_seqs:
                        readable = ''.join([self.idx_to_phonetic.get(idx.item(), '?') for idx in seq])
                        readable_seqs.append(readable)
                    
                    # Store for return
                    self._ctc_decoded_outputs = readable_seqs
                    logger.info(f"CTC decoded sequences: {readable_seqs}")
                
                # Store CTC feature for later use in reranking
                self._ctc_features_for_reranking = reduced_enc_out if self.cfg.ctc_feature_source == "projector" else output['encoder_out']
        
        B, T, D = reduced_enc_out.size()
        instruction = kwargs['source']['text']
        
        # Get instruction embedding with model structure awareness
        try:
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                instruction_embedding = self.decoder.model.model.embed_tokens(instruction)
            elif hasattr(self.decoder, 'model'):
                instruction_embedding = self.decoder.model.embed_tokens(instruction)
            else:
                instruction_embedding = self.decoder.embed_tokens(instruction)
        except AttributeError:
            # Fallback to get_input_embeddings method
            embedding_layer = self.decoder.get_input_embeddings()
            instruction_embedding = embedding_layer(instruction)
            
        llm_input = torch.cat((instruction_embedding, reduced_enc_out), dim=1) 

        # Use a consistent approach for all models
        self.decoder.config.use_cache = True
        
        # Get LLM outputs
        outputs = self.decoder.generate(
            inputs_embeds=llm_input,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            do_sample=False,  # Changed to False to get deterministic outputs
            length_penalty=length_penalty,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=num_beams,  # Important: return multiple sequences
        )
        
        # Store the original sequences for returning with CTC output
        original_sequences = outputs.sequences
        
        # Log the shape of output sequences to debug beam search
        logger.info(f"LLM output sequences shape: {original_sequences.shape}")
        if hasattr(outputs, 'sequences_scores'):
            logger.info(f"LLM sequences_scores shape: {outputs.sequences_scores.shape}")
            logger.info(f"LLM sequences_scores: {outputs.sequences_scores}")
        
        # Store pure LLM output for comparison
        llm_only_output = original_sequences
        
        # Check if reranking with CTC is needed
        if use_ctc_decoding and hasattr(self, '_ctc_features_for_reranking') and ctc_weight_decode is not None and ctc_weight_decode > 0:
            # Get sequences from outputs
            sequences = outputs.sequences
            
            # Convert sequences to list of token lists for each batch item
            batch_size = kwargs['source']['video'].size(0)  # Get actual batch size from input
            num_return_sequences = sequences.size(0) // batch_size if batch_size > 0 else 1
            
            logger.info(f"Actual batch size: {batch_size}, num_return_sequences: {num_return_sequences}")
            
            sequence_lists = [[] for _ in range(batch_size)]
            
            # Group by batch
            for i in range(sequences.size(0)):
                batch_idx = i // num_return_sequences
                if batch_idx < len(sequence_lists):
                    # Skip instruction token IDs which are the same for all beams
                    seq = sequences[i, instruction.size(1):]
                    sequence_lists[batch_idx].append(seq)
            
            # Calculate LLM scores
            llm_scores = []
            for i in range(batch_size):
                if hasattr(outputs, 'sequences_scores'):
                    beam_scores = []
                    for j in range(num_return_sequences):
                        idx = i * num_return_sequences + j
                        if idx < len(outputs.sequences_scores):
                            beam_scores.append(outputs.sequences_scores[idx].item())
                    llm_scores.append(beam_scores)
                else:
                    # Fallback - equal weights if scores not available
                    llm_scores.append([1.0] * len(sequence_lists[i]))
            
            # Log the actual number of beam hypotheses collected
            for i, seqs in enumerate(sequence_lists):
                logger.info(f"BATCH {i}: Collected {len(seqs)} beam hypotheses")
                logger.info(f"BATCH {i}: LLM scores: {llm_scores[i]}")
            
            # If we have empty beam lists or not enough beams, exit early
            if any(len(seqs) == 0 for seqs in sequence_lists) or all(len(seqs) == 1 for seqs in sequence_lists):
                logger.warning("Not enough beam hypotheses collected for reranking, returning LLM output only")
                
                # Return both LLM and CTC outputs if requested
                if return_ctc_outputs and hasattr(self, '_ctc_decoded_outputs'):
                    result = {
                        'llm_output': llm_only_output,
                        'ctc_output': self._ctc_decoded_outputs,
                        'hybrid_output': None  # No hybrid reranking performed
                    }
                    delattr(self, '_ctc_decoded_outputs')
                    return result
                
                return llm_only_output
            
            # Perform hybrid reranking
            reranked_indices = self.hybrid_reranking(
                self._ctc_features_for_reranking,
                llm_scores,
                sequence_lists,
                ctc_weight=ctc_weight_decode,
                feature_source=self.cfg.ctc_feature_source
            )
            
            # Reorder sequences based on reranking
            reranked_sequences = []
            for b in range(batch_size):
                # Get the top reranked sequence
                top_idx = reranked_indices[b][0]
                original_idx = b * num_return_sequences + top_idx
                reranked_sequences.append(sequences[original_idx])
                
                # For debugging: log the selected sequence after reranking
                if hasattr(self.decoder, 'tokenizer'):
                    tokenizer = self.decoder.tokenizer
                    logger.info(f"BATCH {b}: SELECTED OUTPUT AFTER RERANKING (index {top_idx}):")
                    decoded = tokenizer.decode(sequences[original_idx], skip_special_tokens=True)
                    logger.info(f"  {decoded}")
            
            # Return reranked sequences
            if len(reranked_sequences) == 1:
                # For comparison with the original
                if hasattr(self.decoder, 'tokenizer') and batch_size == 1:
                    tokenizer = self.decoder.tokenizer
                    orig_decoded = tokenizer.decode(sequences[0], skip_special_tokens=True)
                    new_decoded = tokenizer.decode(reranked_sequences[0], skip_special_tokens=True)
                    logger.info(f"ORIGINAL TOP: {orig_decoded}")
                    logger.info(f"RERANKED TOP: {new_decoded}")
                    logger.info(f"CHANGED: {orig_decoded != new_decoded}")
                
                # If requested, return both LLM and hybrid outputs
                if return_both_outputs:
                    result = {
                        'llm_output': llm_only_output,
                        'hybrid_output': reranked_sequences[0],
                        'ctc_output': self._ctc_decoded_outputs if hasattr(self, '_ctc_decoded_outputs') else None
                    }
                    if hasattr(self, '_ctc_decoded_outputs'):
                        delattr(self, '_ctc_decoded_outputs')
                    return result
                
                return reranked_sequences[0]
            else:
                # If requested, return both LLM and hybrid outputs
                if return_both_outputs:
                    result = {
                        'llm_output': llm_only_output,
                        'hybrid_output': torch.stack(reranked_sequences),
                        'ctc_output': self._ctc_decoded_outputs if hasattr(self, '_ctc_decoded_outputs') else None
                    }
                    if hasattr(self, '_ctc_decoded_outputs'):
                        delattr(self, '_ctc_decoded_outputs')
                    return result
                
                return torch.stack(reranked_sequences)
            
        # Clear temporary storage
        if hasattr(self, '_ctc_features_for_reranking'):
            delattr(self, '_ctc_features_for_reranking')

        # Return both LLM and CTC outputs if requested
        if return_ctc_outputs and hasattr(self, '_ctc_decoded_outputs'):
            # Create a structure to return both
            result = {
                'llm_output': outputs.sequences,
                'ctc_output': self._ctc_decoded_outputs,
                'hybrid_output': None  # No hybrid reranking performed
            }
            # Clean up temporary storage
            delattr(self, '_ctc_decoded_outputs')
            return result
        
        return outputs.sequences

    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """
        Standard Fairseq hook called by the trainer to update model state.
        This is the idiomatic place to ensure training state consistency.
        """
        # Call parent method
        super().set_num_updates(num_updates) 
        
        # Store updates counter for use in model logic
        self.num_updates = num_updates
        
        # Ensure CTC loss is used if gradients are enabled
        # Use a custom attribute to track our intent rather than relying on self.training
        self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # Log when set_num_updates is called (helps debugging frequency)
        logger.info(f"set_num_updates called with update {num_updates}, grad_enabled={torch.is_grad_enabled()}")
        
        # Log CTC status (not tied to batch counter to ensure visibility)
        logger.info(f"CTC usage status at update {num_updates}: {self._use_ctc_in_forward} (training={self.training}, grad_enabled={torch.is_grad_enabled()})")
        
        return self.num_updates

    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if 'lora' in k or 'avfeat_to_llm' in k or 'encoder' in k or 'ctc_head' in k}
        return state

    def calculate_ctc_scores(self, encoder_features, sequences, feature_source="projector"):
        """
        Calculate CTC scores for a set of sequences for use in reranking.
        
        Args:
            encoder_features: tensor of shape [B, T, D] (based on feature_source)
            sequences: list of token sequences to score
            feature_source: whether to use "encoder" or "projector" features
            
        Returns:
            scores: tensor of CTC scores for each sequence
        """
        # Skip if CTC not available
        if feature_source == "encoder" and self.ctc_head_encoder is None:
            logger.warning("Requested encoder CTC scoring but encoder CTC head is not available")
            return None
        elif feature_source == "projector" and self.ctc_head_projector is None:
            logger.warning("Requested projector CTC scoring but projector CTC head is not available")
            return None
        
        logger.info(f"Calculating CTC scores using {feature_source} features")
        
        # Get logits from appropriate CTC head
        with torch.no_grad():
            if feature_source == "encoder":
                ctc_features_matched = encoder_features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
            else:
                ctc_features_matched = encoder_features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
            
            # Apply log softmax to CTC logits
            log_probs = F.log_softmax(ctc_logits, dim=-1)  # [B, T, V]
        
        # Calculate scores for each sequence
        batch_size = log_probs.size(0)
        input_lengths = torch.full((batch_size,), log_probs.size(1), device=log_probs.device, dtype=torch.long)
        scores = []
        
        logger.info(f"CTC log_probs shape: {log_probs.shape}, device: {log_probs.device}")
        logger.info(f"Number of batch items: {batch_size}, sequences per batch: {[len(s) for s in sequences]}")
        
        for b in range(batch_size):
            batch_scores = []
            batch_log_probs = log_probs[b].unsqueeze(0)  # [1, T, V]
            
            # Process each sequence
            for seq in sequences[b]:
                try:
                    # Map sequence to phonetic indices for scoring
                    # This is a simplified approach - ideally we would do proper mapping
                    phonetic_target = torch.zeros_like(seq)
                    
                    # Fill with known indices or defaults
                    for i, token_id in enumerate(seq):
                        # Skip padding tokens
                        if token_id == 0:
                            continue
                        
                        # Try to map to a phonetic token (simplified)
                        # In a real implementation, this would use a proper mapping
                        phonetic_target[i] = self.phonetic_vocab.get('<unk>', 1)  # Default to <unk> if not found
                    
                    # Calculate target length (non-padding tokens)
                    target_length = (phonetic_target != 0).sum().item()
                    
                    # Skip empty targets
                    if target_length == 0:
                        batch_scores.append(0.0)
                        continue
                    
                    # Make sure target length doesn't exceed input length
                    if target_length > input_lengths[0].item():
                        logger.warning(f"Target length {target_length} exceeds input length {input_lengths[0].item()}, reducing")
                        target_length = input_lengths[0].item()
                    
                    # Prepare target tensor
                    target = phonetic_target[:target_length].unsqueeze(0)  # [1, L]
                    target_lengths = torch.tensor([target_length], device=target.device, dtype=torch.long)
                    
                    # CTC forward-backward algorithm to get probability
                    batch_log_probs_float = batch_log_probs.float()  # Ensure float32 for CTC calculation
                    loss = F.ctc_loss(
                        batch_log_probs_float.transpose(0, 1),  # [T, 1, V]
                        target,
                        input_lengths,
                        target_lengths,
                        blank=self.cfg.ctc_blank_idx,
                        reduction='mean',
                        zero_infinity=True
                    )
                    
                    # Convert loss to score (negative loss)
                    score = -loss.item()
                    
                    # Debug logging
                    logger.info(f"CTC score for sequence: {score:.4f}, length: {target_length}")
                    
                    batch_scores.append(score)
                except Exception as e:
                    logger.error(f"Error calculating CTC score: {e}")
                    # Default to a very low score
                    batch_scores.append(-1000.0)
            
            scores.append(batch_scores)
        
        logger.info(f"Final CTC scores: {scores}")
        return scores
        
    def hybrid_reranking(self, encoder_features, llm_scores, sequences, ctc_weight=0.3, feature_source="projector"):
        """
        Rerank sequences using a combination of LLM and CTC scores
        
        Args:
            encoder_features: tensor of encoder features
            llm_scores: list of LLM scores for sequences
            sequences: list of sequences to rerank
            ctc_weight: weight for CTC scores (1-ctc_weight for LLM scores)
            feature_source: whether to use "encoder" or "projector" features
            
        Returns:
            reranked_indices: list of reranked indices for each batch item
        """
        logger.info(f"HYBRID RERANKING: Using CTC weight {ctc_weight}")
        
        # Calculate CTC scores
        ctc_scores = self.calculate_ctc_scores(encoder_features, sequences, feature_source)
        
        if ctc_scores is None:
            # Fallback to original rankings if CTC scoring failed
            logger.warning("HYBRID RERANKING: CTC scoring failed, using original order")
            return list(range(len(llm_scores[0])))
        
        # Combine scores
        reranked_indices = []
        for b in range(len(llm_scores)):
            # Log original ranking
            logger.info(f"BATCH {b}: Original LLM scores: {llm_scores[b]}")
            logger.info(f"BATCH {b}: CTC scores: {ctc_scores[b]}")
            
            # Normalize scores
            llm_scores_norm = torch.tensor(llm_scores[b])
            llm_scores_norm = (llm_scores_norm - llm_scores_norm.mean()) / (llm_scores_norm.std() + 1e-9)
            
            ctc_scores_norm = torch.tensor(ctc_scores[b])
            ctc_scores_norm = (ctc_scores_norm - ctc_scores_norm.mean()) / (ctc_scores_norm.std() + 1e-9)
            
            # Combine scores
            combined_scores = (1 - ctc_weight) * llm_scores_norm + ctc_weight * ctc_scores_norm
            
            # Get indices of sorted combined scores
            _, indices = torch.sort(combined_scores, descending=True)
            reranked_indices.append(indices.tolist())
            
            # Log reranking results
            logger.info(f"BATCH {b}: Normalized LLM scores: {llm_scores_norm.tolist()}")
            logger.info(f"BATCH {b}: Normalized CTC scores: {ctc_scores_norm.tolist()}")
            logger.info(f"BATCH {b}: Combined scores: {combined_scores.tolist()}")
            logger.info(f"BATCH {b}: Reranked indices: {indices.tolist()}")
        
        return reranked_indices


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@register_model("vsp_llm_ctc", dataclass=VSPLLMConfig)
class VSP_LLM_With_CTC(avhubert_llm_seq2seq_cluster_count):
    """VSP-LLM with auxiliary CTC loss"""
    
    def __init__(self, encoder, decoder, cfg):
        super().__init__(encoder, decoder, cfg)
        
        # Initialize num_updates for logging
        self.num_updates = 0
        self.batch_counter = 0  # Add batch counter for more frequent logging
        self._use_ctc_in_forward = cfg.use_ctc  # Add tracking attribute for CTC usage
        
        # Get hidden size dynamically from the decoder's configuration
        # This will be the output dimension of our projector
        projector_out_dim = None
        
        # Method 1: Get directly from decoder config
        if hasattr(decoder, 'config'):
            if hasattr(decoder.config, 'hidden_size'):
                projector_out_dim = decoder.config.hidden_size
                logger.info(f"Found projector output dimension from decoder hidden_size: {projector_out_dim}")
            elif hasattr(decoder.config, 'n_embd'):  # For GPT style models
                projector_out_dim = decoder.config.n_embd
                logger.info(f"Found projector output dimension from decoder n_embd: {projector_out_dim}")
        
        # Method 2: Try to infer from the decoder embeddings
        if projector_out_dim is None and hasattr(decoder, 'get_input_embeddings'):
            embedding_weight = decoder.get_input_embeddings().weight
            if hasattr(embedding_weight, 'shape') and len(embedding_weight.shape) > 1:
                projector_out_dim = embedding_weight.shape[1]
                logger.info(f"Inferred projector output dimension from embedding weight: {projector_out_dim}")
        
        # Method 3: If all else fails, try to find from model structure
        if projector_out_dim is None:
            # Check different model architectures
            if hasattr(decoder, 'model') and hasattr(decoder.model, 'model'):
                if hasattr(decoder.model.model, 'embed_tokens'):
                    projector_out_dim = decoder.model.model.embed_tokens.weight.shape[1]
                    logger.info(f"Found projector dimension from nested model.model.embed_tokens: {projector_out_dim}")
            elif hasattr(decoder, 'model'):
                if hasattr(decoder.model, 'embed_tokens'):
                    projector_out_dim = decoder.model.embed_tokens.weight.shape[1]
                    logger.info(f"Found projector dimension from model.embed_tokens: {projector_out_dim}")
            elif hasattr(decoder, 'embed_tokens'):
                projector_out_dim = decoder.embed_tokens.weight.shape[1]
                logger.info(f"Found projector dimension from embed_tokens: {projector_out_dim}")
        
        # Safety check - we must have a valid dimension at this point
        if projector_out_dim is None:
            # Last resort: Use the dimension from our own projector
            if hasattr(self.avfeat_to_llm, 'output_dim'):
                projector_out_dim = self.avfeat_to_llm.output_dim
                logger.info(f"Using projector's declared output_dim: {projector_out_dim}")
            else:
                raise ValueError("Could not determine projector output dimension. Please check model configuration.")
        
        # Get encoder output dimension for encoder-based CTC
        encoder_dim = None
        
        # Method 1: Try to infer from the encoder output size
        if hasattr(self.encoder.w2v_model, 'encoder_embed_dim'):
            encoder_dim = self.encoder.w2v_model.encoder_embed_dim
            logger.info(f"Found encoder_dim from encoder_embed_dim: {encoder_dim}")
        
        # Method 2: Try common architecture patterns
        elif hasattr(self.encoder.w2v_model, 'encoder') and hasattr(self.encoder.w2v_model.encoder, 'layernorm'):
            encoder_dim = self.encoder.w2v_model.encoder.layernorm.weight.size(0)
            logger.info(f"Found encoder_dim from encoder.layernorm: {encoder_dim}")
        
        # Method 3: Try accessing a different attribute path
        elif hasattr(self.encoder.w2v_model, 'encoder') and hasattr(self.encoder.w2v_model.encoder, 'layers') and len(self.encoder.w2v_model.encoder.layers) > 0:
            # Get from the last layer's output dimension
            last_layer = self.encoder.w2v_model.encoder.layers[-1]
            if hasattr(last_layer, 'self_attn') and hasattr(last_layer.self_attn, 'out_proj'):
                encoder_dim = last_layer.self_attn.out_proj.out_features
                logger.info(f"Found encoder_dim from encoder layers: {encoder_dim}")
        
        # Fallback to a hardcoded value only if all else fails
        if encoder_dim is None:
            # Use a reasonable default based on common encoder dimensions
            encoder_dim = 1024  # Common AV-HuBERT dimension
            logger.warning(f"Could not determine encoder dimension, using default: {encoder_dim}")
        
        # Define a phonetic vocabulary for CTC
        # Create a phonetic vocabulary of the specified size
        self.ctc_vocab_size = getattr(cfg, 'ctc_vocab_size', 50)
        
        # Create a dict mapping characters to indices
        # Start with a blank token at index 0
        self.phonetic_vocab = {'<blank>': 0}
        
        # Add basic phonetic symbols (vowels, consonants, etc.)
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
        special = ['<unk>', '<pad>', '<s>', ' ', ' ']
        
        # Add these symbols to our vocabulary
        all_symbols = vowels + consonants + special
        
        # Ensure we don't exceed the vocabulary size
        available_symbols = all_symbols[:self.ctc_vocab_size - 1]  # -1 for blank token
        
        # Add symbols to the vocabulary
        for i, symbol in enumerate(available_symbols):
            self.phonetic_vocab[symbol] = i + 1  # +1 because blank is 0
        
        # Create reverse mapping
        self.idx_to_phonetic = {idx: char for char, idx in self.phonetic_vocab.items()}
        
        # Set actual vocab size based on created vocabulary
        self.ctc_vocab_size = len(self.phonetic_vocab)
        
        # Use this vocab size for the CTC head
        vocab_size = self.ctc_vocab_size
        
        # Sanity check: ensure vocab size is reasonable for phonetic vocabulary
        if vocab_size > 100:
            logger.warning(f"Phonetic vocabulary size {vocab_size} is unusually large. Check configuration.")
            # Force reasonable size for phonetic vocabulary
            logger.warning(f"Forcing phonetic vocabulary size to 50 (was {vocab_size})")
            vocab_size = 50
            self.ctc_vocab_size = vocab_size
        
        logger.info(f"Using phonetic vocabulary for CTC with {vocab_size} tokens")
        
        # Log the determined dimensions with clearer information
        logger.info(f"Encoder CTC head will use dimensions: {encoder_dim} → {vocab_size} (phonetic vocab)")
        logger.info(f"Projector CTC head will use dimensions: {projector_out_dim} → {vocab_size} (phonetic vocab)")
        
        # Add structured CTC heads for both feature sources
        # Use a more sophisticated architecture with multiple layers for better mapping
        if cfg.ctc_feature_source == "encoder":
            # Initialize the encoder-based CTC head (only if we're using it)
            hidden_dim = min(encoder_dim, 512)  # Use smaller hidden dimension for efficiency
            self.ctc_head_encoder = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            )
            # Initialize only the projector if we're using it
            self.ctc_head_projector = None
        else:
            # Initialize the projector-based CTC head (only if we're using it)
            hidden_dim = min(projector_out_dim, 512)  # Use smaller hidden dimension for efficiency
            self.ctc_head_projector = nn.Sequential(
                nn.Linear(projector_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            )
            # Initialize only the encoder if we're using it
            self.ctc_head_encoder = None
        
        logger.info(f"Using structured CTC head with hidden dimension: {hidden_dim}")
        
        # Auto-set blank index to avoid conflict with padding token
        if not hasattr(cfg, 'ctc_blank_idx') or cfg.ctc_blank_idx == 0 or cfg.ctc_blank_idx is None or (isinstance(cfg.ctc_blank_idx, str) and not cfg.ctc_blank_idx.strip()):
            # For phonetic vocab, blank is already at index 0
            self.cfg.ctc_blank_idx = 0
            logger.info(f"Using phonetic vocabulary with blank_idx=0")
            
            # Try to print the actual token if tokenizer is available
            if hasattr(decoder, 'tokenizer'):
                tokenizer = decoder.tokenizer
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    logger.info(f"CTC blank_token: {tokenizer.pad_token}")
        
        # Log CTC configuration
        logger.info(f"==========================================================")
        logger.info(f"INITIALIZING MODEL WITH CTC: weight={cfg.ctc_weight}, blank={cfg.ctc_blank_idx}")
        logger.info(f"CTC feature source: {cfg.ctc_feature_source}")
        logger.info(f"==========================================================")

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        # Use the parent class to create the base model
        base_model = super().build_model(cfg, task)
        
        # Add CTC weight decode param if not present
        if not hasattr(cfg, 'ctc_weight_decode'):
            cfg.ctc_weight_decode = getattr(cfg, 'ctc_weight', 0.3)
        
        # Create our CTC-enabled version
        return cls(base_model.encoder, base_model.decoder, cfg)
    
    def forward(self, **kwargs):
        # Increment batch counter for logging
        self.batch_counter += 1
        
        # Always update CTC status at the start of forward to ensure it's correct
        # This prevents issues when transitioning between train and validation
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'use_ctc'):
            self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # First get encoder output
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        
        # Store raw encoder output for CTC loss if needed for comparison
        raw_encoder_out = output['encoder_out'].clone()
        
        # Continue with normal forward pass
        # Get the original size before applying projector
        orig_seq_len = output['encoder_out'].size(1)
        
        # Get transcript information for cross-modal alignment
        labels = kwargs['target_list'].clone()
        
        # Get labels embedding directly from the model
        if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
            labels_embedding = self.decoder.model.model.embed_tokens(labels)
        elif hasattr(self.decoder, 'model'):
            labels_embedding = self.decoder.model.embed_tokens(labels)
        else:
            labels_embedding = self.decoder.embed_tokens(labels)
        
        # Simplified rule for text-based projectors:
        # Linear and MLP are cluster-based, everything else is text-based
        is_text_based = self.cfg.projector_type.lower() not in ['linear', 'mlp']
        
        # Apply projector based on type
        alignment_loss = torch.tensor(0.0, device=output['encoder_out'].device)
        
        if is_text_based:
            # For all text-based projectors, always pass transcript tokens embeddings
            if not self.logged_projector_shape:
                logger.info(f"Using transcript tokens for vision-text mapping in {self.cfg.projector_type}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (labels != 0)
            
            # Convert embeddings to match encoder output dtype
            labels_embedding_matched = labels_embedding.to(dtype=output['encoder_out'].dtype)
            
            # Check projector's forward method signature to determine which parameters to pass
            forward_params = inspect.signature(self.avfeat_to_llm.forward).parameters
            
            # Base parameters that all projectors accept
            projector_args = {
                "x": output['encoder_out']
            }
            
            # Add text_tokens parameter if it exists in the signature
            if "text_tokens" in forward_params:
                projector_args["text_tokens"] = labels_embedding_matched
                
            # Add text_mask parameter if it exists in the signature
            if "text_mask" in forward_params:
                projector_args["text_mask"] = text_mask
                
            # For QFormer-specific implementations that use text_embeddings instead of text_tokens
            if "text_embeddings" in forward_params and "text_tokens" not in forward_params:
                projector_args["text_embeddings"] = labels_embedding_matched
                
            # Pass the appropriate parameters to the projector
            output['encoder_out'] = self.avfeat_to_llm(**projector_args)
        else:
            # For non-text-based projectors (linear, mlp), just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        
        # Check if we're using a query-based projector that changes sequence length
        proj_seq_len = output['encoder_out'].size(1)
        
        # Query-based projectors are the same as text-based in our case, 
        # since only linear and mlp are not query-based
        is_query_based = is_text_based
            
        # Handle different projector types
        if is_query_based:
            # For query-based projectors that return fixed number of tokens
            # We skip the cluster-based aggregation as these projectors already aggregate
            reduced_enc_out = output['encoder_out']
            # Only log the shape once
            if not self.logged_projector_shape:
                logger.info(f"Using query-based projector output with shape: {reduced_enc_out.size()}")
                self.logged_projector_shape = True
        else:
            # For projectors that maintain sequence length, use cluster-based aggregation
            cluster_counts = kwargs['source']['cluster_counts'][0]  # tensor list
            
            results_tensor = []
            start_idx = 0
            for clutser_num in cluster_counts:
                end_idx = start_idx + clutser_num
                slice = output['encoder_out'][:,start_idx:end_idx,:]
                
                # Original method: simple mean aggregation
                mean_avg = slice.mean(dim=1, keepdim=True)
                results_tensor.append(mean_avg)
                
                start_idx = end_idx
    
            # Verify we processed the entire sequence
            assert(cluster_counts.sum().item() == proj_seq_len)
            reduced_enc_out = torch.cat(results_tensor, dim=1)
            # Only log the shape once
            if not self.logged_projector_shape:
                logger.info(f"Using mean cluster aggregation with output shape: {reduced_enc_out.size()}")
                self.logged_projector_shape = True
        
        B, T, D = reduced_enc_out.size()
        instruction = kwargs['source']['text']
        
        # Get instruction embedding with model structure awareness
        try:
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                instruction_embedding = self.decoder.model.model.embed_tokens(instruction)
            elif hasattr(self.decoder, 'model'):
                instruction_embedding = self.decoder.model.embed_tokens(instruction)
            else:
                instruction_embedding = self.decoder.embed_tokens(instruction)
        except AttributeError:
            # Fallback to get_input_embeddings method
            embedding_layer = self.decoder.get_input_embeddings()
            instruction_embedding = embedding_layer(instruction)

        llm_input = torch.cat((instruction_embedding, reduced_enc_out, labels_embedding), dim=1)
        
        # Prepare labels for loss calculation (used by all model types)
        llm_labels = labels.clone()
        llm_labels[llm_labels == 0] = -100
        
        _, instruction_embedding_t, _ = instruction_embedding.size()
        target_ids = torch.full((B, T + instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)
        
        # Use a single code path for all models instead of model-specific branches
        llm_out = self.decoder(inputs_embeds=llm_input, labels=llm_labels, return_dict=True)
        lm_loss = llm_out.loss
        
        # Calculate CTC loss if enabled and in training mode
        total_loss = lm_loss
        if self._use_ctc_in_forward:
            # Initialize input_lengths to None to avoid UnboundLocalError
            input_lengths = None
            
            # Get CTC features based on the configured source
            if self.cfg.ctc_feature_source == "encoder":
                # Use raw encoder output for CTC
                ctc_features = raw_encoder_out.clone()
                
                # Only calculate logits if using encoder features
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
                
                # For encoder output, be careful with input lengths
                if hasattr(self, 'encoder_outputs_per_layer'):
                    # Use encoder-specific information if available
                    input_lengths = self.encoder_outputs_per_layer.get('lengths', None)
                
                if input_lengths is None:
                    # Fallback to assuming all time steps are valid
                    input_lengths = torch.full((raw_encoder_out.size(0),), 
                                            raw_encoder_out.size(1),
                                            device=raw_encoder_out.device,
                                            dtype=torch.long)
                
                # Log which feature source is being used
                if self.batch_counter % 20 == 0:
                    logger.info(f"Using raw encoder output for CTC, shape: {ctc_features.shape}")
            else:
                # Use projector output for CTC (default)
                ctc_features = reduced_enc_out.clone()
                ctc_features_matched = ctc_features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
                
                # For projector output, be more careful with input lengths
                # Key difference: projector outputs might have different sequence length characteristics
                # compared to encoder outputs, especially after clustering/aggregation
                if is_query_based:
                    # For query-based projectors, each output token should be considered
                    input_lengths = torch.full((ctc_features.size(0),), 
                                          ctc_features.size(1),
                                          device=ctc_features.device,
                                          dtype=torch.long)
                else:
                    # For clustering-based approaches, the sequence reflects the number of clusters
                    # which should match the number of distinct segments in the input
                    input_lengths = torch.full((ctc_features.size(0),), 
                                          ctc_features.size(1),
                                          device=ctc_features.device,
                                          dtype=torch.long)
                
                # Log which feature source is being used
                if self.batch_counter % 20 == 0:
                    logger.info(f"Using projector output for CTC, shape: {ctc_features.shape}")
                
            # Apply log softmax to CTC logits
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
            
            # Map target to phonetic indices
            # Create mapping tensor for LLM token IDs to phonetic IDs (map everything to <unk>)
            # In a real implementation, you would have a more sophisticated mapping
            # For now, we'll just use a simple strategy for testing
            unk_idx = self.phonetic_vocab.get('<unk>', 1)  # Default to 1 if not found
            
            # Clone target to avoid modifying the original
            phonetic_target = torch.full_like(labels, unk_idx)
            
            # Get non-padding tokens
            non_pad_mask = (labels != 0)
            
            # Set non-pad tokens to unk_idx (this is a simplified approach)
            # In a real implementation, you would map each token to an appropriate phonetic sequence
            phonetic_target = phonetic_target * non_pad_mask
            
            # Use phonetic_target for CTC loss calculation
            target = phonetic_target
            
            # Remove padding tokens for CTC loss
            pad_mask = (target != 0)  # 0 is padding token
            target_lengths = pad_mask.sum(-1)
            
            # Add diagnostic logs to help debug CTC issues
            if self.batch_counter % 20 == 0:
                non_zero_targets = (target_lengths > 0).sum().item()
                total_targets = target_lengths.size(0)
                logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                logger.info(f"CTC input_lengths: min={input_lengths.min().item()}, max={input_lengths.max().item()}")
                logger.info(f"CTC target_lengths: min={target_lengths.min().item()}, max={target_lengths.max().item()}")
                
                # Check if we have invalid CTC constraints (input shorter than target)
                invalid_lens = (input_lengths < target_lengths).sum().item()
                if invalid_lens > 0:
                    logger.warning(f"CTC constraint violation: {invalid_lens} sequences have input_length < target_length")
                
                # Log vocabulary information
                logger.info(f"Using phonetic vocabulary for CTC targets, size: {self.ctc_vocab_size}")
            
            # Calculate CTC loss
            try:
                # Check for empty targets or input
                if (target_lengths == 0).all() or (input_lengths == 0).all():
                    logger.warning(f"Empty CTC targets or inputs detected - CTC loss will be ignored")
                    ctc_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
                else:
                    # CTC needs float32 for loss calculation
                    log_probs_float = log_probs.float()
                    ctc_loss = F.ctc_loss(
                        log_probs_float,
                        target,
                        input_lengths,
                        target_lengths,
                        blank=self.cfg.ctc_blank_idx,
                        reduction="mean",
                        zero_infinity=True,
                    )
                    
                    # Check if loss is suspiciously close to zero
                    if ctc_loss.item() < 1e-6:
                        logger.warning(f"CTC loss is suspiciously close to zero: {ctc_loss.item()}")
                        
                        # Check logits for possible numerical issues
                        max_logit = ctc_logits.max().item()
                        min_logit = ctc_logits.min().item()
                        mean_logit = ctc_logits.mean().item()
                        logger.warning(f"CTC logits stats: min={min_logit:.4f}, max={max_logit:.4f}, mean={mean_logit:.4f}")
                
                # Combine losses with weighting
                # Convert to same dtype for combination
                # Add alignment loss if available (for cross-modal learning)
                if is_text_based and alignment_loss.item() > 0:
                    # Add alignment loss with a weight (e.g., 0.1)
                    alignment_weight = 0.1
                    total_loss = (1 - self.cfg.ctc_weight - alignment_weight) * lm_loss + \
                                self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype) + \
                                alignment_weight * alignment_loss.to(lm_loss.dtype)
                    
                    # Log all losses
                    if self.batch_counter % 20 == 0:
                        lm_loss_val = float(lm_loss.detach().cpu().item())
                        ctc_loss_val = float(ctc_loss.detach().cpu().item())
                        align_loss_val = float(alignment_loss.detach().cpu().item())
                        logger.info(f"Batch {self.batch_counter}: LM loss: {lm_loss_val:.4f}, "
                                   f"CTC loss: {ctc_loss_val:.4f}, "
                                   f"Alignment loss: {align_loss_val:.4f}")
                else:
                    # Standard weighting without alignment loss
                    total_loss = (1 - self.cfg.ctc_weight) * lm_loss + self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype)
                    
                    # Log losses occasionally
                    if self.batch_counter % 20 == 0:
                        lm_loss_val = float(lm_loss.detach().cpu().item())
                        ctc_loss_val = float(ctc_loss.detach().cpu().item())
                        logger.info(f"Batch {self.batch_counter}: LM loss: {lm_loss_val:.4f}, CTC loss: {ctc_loss_val:.4f}")
                
            except Exception as e:
                logger.warning(f"CTC loss calculation failed: {e}")
                # Add detailed exception information for debugging
                import traceback
                logger.warning(f"CTC exception traceback: {traceback.format_exc()}")
                logger.warning(f"CTC input shape: {ctc_features.shape}, target shape: {target.shape}")
                logger.warning(f"input_lengths: {input_lengths}")
                logger.warning(f"target_lengths: {target_lengths}")
                total_loss = lm_loss  # Fallback to just LM loss
        
        # Return combined loss, logits, and labels
        return total_loss, llm_out.logits, llm_labels

    def set_num_updates(self, num_updates):
        """
        Standard Fairseq hook called by the trainer to update model state.
        This is the idiomatic place to ensure training state consistency.
        """
        # Call parent method
        super().set_num_updates(num_updates) 
        
        # Store updates counter for use in model logic
        self.num_updates = num_updates
        
        # Ensure CTC loss is used if gradients are enabled
        # Use a custom attribute to track our intent rather than relying on self.training
        self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # Log when set_num_updates is called (helps debugging frequency)
        logger.info(f"set_num_updates called with update {num_updates}, grad_enabled={torch.is_grad_enabled()}")
        
        # Log CTC status (not tied to batch counter to ensure visibility)
        logger.info(f"CTC usage status at update {num_updates}: {self._use_ctc_in_forward} (training={self.training}, grad_enabled={torch.is_grad_enabled()})")
        
        return self.num_updates

    def ctc_greedy_decode(self, features, feature_source="projector"):
        """
        Performs greedy CTC decoding on the given features.
        
        Args:
            features: Tensor of shape [B, T, D] from encoder or projector
            feature_source: Whether to use "encoder" or "projector" CTC head
            
        Returns:
            List of decoded sequences as strings (if phonetic vocab) or token indices
        """
        # Skip if CTC not available
        if feature_source == "encoder" and self.ctc_head_encoder is None:
            logger.warning("Requested encoder CTC decoding but encoder CTC head is not available")
            return None
        elif feature_source == "projector" and self.ctc_head_projector is None:
            logger.warning("Requested projector CTC decoding but projector CTC head is not available")
            return None
            
        # Get logits from appropriate CTC head
        with torch.no_grad():
            if feature_source == "encoder":
                features_matched = features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                logits = self.ctc_head_encoder(features_matched)
            else:
                features_matched = features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                logits = self.ctc_head_projector(features_matched)
            
            # Get most likely tokens at each position
            best_paths = torch.argmax(logits, dim=-1)  # [B, T]
        
        # Apply CTC decoding rules (merge repeated tokens, remove blanks)
        batch_size = best_paths.size(0)
        decoded_seqs = []
        
        for b in range(batch_size):
            path = best_paths[b]
            decoded = []
            prev = -1  # Different from all valid indices
            
            for token_idx in path:
                token = token_idx.item()
                if token != prev and token != self.cfg.ctc_blank_idx:
                    decoded.append(token)
                prev = token
            
            # Convert to string if using phonetic vocabulary
            if hasattr(self, 'idx_to_phonetic'):
                decoded_str = ''.join([self.idx_to_phonetic.get(idx, '?') for idx in decoded])
                decoded_seqs.append(decoded_str)
            else:
                # Return indices
                decoded_seqs.append(decoded)
        
        return decoded_seqs
    
    def ctc_beam_decode(self, features, beam_size=10, feature_source="projector"):
        """
        Performs beam search CTC decoding on the given features.
        
        Args:
            features: Tensor of shape [B, T, D] from encoder or projector
            beam_size: Beam size for the search
            feature_source: Whether to use "encoder" or "projector" CTC head
            
        Returns:
            List of decoded sequences as strings (if phonetic vocab) or token indices
        """
        # Skip if CTC not available
        if feature_source == "encoder" and self.ctc_head_encoder is None:
            logger.warning("Requested encoder CTC beam decoding but encoder CTC head is not available")
            return None
        elif feature_source == "projector" and self.ctc_head_projector is None:
            logger.warning("Requested projector CTC beam decoding but projector CTC head is not available")
            return None
            
        # Get logits from appropriate CTC head
        with torch.no_grad():
            if feature_source == "encoder":
                features_matched = features.to(dtype=next(self.ctc_head_encoder.parameters()).dtype)
                logits = self.ctc_head_encoder(features_matched)
            else:
                features_matched = features.to(dtype=next(self.ctc_head_projector.parameters()).dtype)
                logits = self.ctc_head_projector(features_matched)
            
            # Apply log softmax
            log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        
        # Simple beam search implementation
        batch_size = log_probs.size(0)
        vocab_size = log_probs.size(2)
        blank_idx = self.cfg.ctc_blank_idx
        
        decoded_seqs = []
        
        for b in range(batch_size):
            # Initialize with empty sequence and score 0
            beams = [{'sequence': [], 'score': 0.0, 'last_token': -1}]
            
            # Process each timestep
            for t in range(log_probs.size(1)):
                new_beams = []
                
                # For each existing beam
                for beam in beams:
                    # Get log probabilities for this timestep
                    t_log_probs = log_probs[b, t]
                    
                    # Try extending with each token
                    for v in range(vocab_size):
                        # Skip if same as last token or blank
                        if v == beam['last_token'] or v == blank_idx:
                            continue
                        
                        # Create new beam with this token
                        new_sequence = beam['sequence'] + [v]
                        new_score = beam['score'] + t_log_probs[v].item()
                        new_beams.append({
                            'sequence': new_sequence,
                            'score': new_score,
                            'last_token': v
                        })
                    
                    # Try the blank token as well (keeps sequence unchanged)
                    new_beams.append({
                        'sequence': beam['sequence'],
                        'score': beam['score'] + t_log_probs[blank_idx].item(),
                        'last_token': beam['last_token']
                    })
                
                # Sort by score and keep top-k
                beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
            
            # Get best beam
            best_beam = beams[0]['sequence']
            
            # Convert to string if using phonetic vocabulary
            if hasattr(self, 'idx_to_phonetic'):
                decoded_str = ''.join([self.idx_to_phonetic.get(idx, '?') for idx in best_beam])
                decoded_seqs.append(decoded_str)
            else:
                # Return indices
                decoded_seqs.append(best_beam)
        
        return decoded_seqs

    def print_outputs(self, tokenizer, result):
        """
        Print both CTC and LLM outputs in a readable format
        
        Args:
            tokenizer: The tokenizer to decode LLM outputs
            result: The result from generate() containing both outputs
        """
        if isinstance(result, dict) and 'llm_output' in result and 'ctc_output' in result:
            # Print CTC output
            print("\n===== CTC Output =====")
            for i, ctc_seq in enumerate(result['ctc_output']):
                print(f"Sample {i+1}: {ctc_seq}")
            
            # Print LLM output
            print("\n===== LLM Output =====")
            for i in range(result['llm_output'].size(0)):
                decoded = tokenizer.decode(result['llm_output'][i], skip_special_tokens=True)
                print(f"Sample {i+1}: {decoded}")
        else:
            # Handle case where only LLM output is available
            print("\n===== LLM Output Only =====")
            if hasattr(result, 'size'):  # Tensor output
                for i in range(result.size(0)):
                    decoded = tokenizer.decode(result[i], skip_special_tokens=True)
                    print(f"Sample {i+1}: {decoded}")
            else:
                print("No recognizable output format")