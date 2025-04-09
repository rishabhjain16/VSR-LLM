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
    ctc_use_char_level: bool = field(
        default=True,
        metadata={"help": "Use character-level tokenization for CTC instead of word-level (recommended)"}
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
        metadata={"help": "Index to use for blank token in CTC (if 0 or empty, will use last token in vocabulary)"}
    )
    ctc_feature_source: str = field(
        default="projector",
        metadata={"help": "Source of features for CTC loss: 'encoder' (raw AV-HuBERT) or 'projector' (after projection)"}
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
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'use_ctc'):
            self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # First get encoder output
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        
        # Store raw encoder output for CTC loss if needed
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
                projector_args["text_tokens"] = labels  # Pass raw token IDs for BLIP2 and Comprehensive QFormer
                # Log for debugging - remove later
                logger.info(f"Passing text_tokens to {self.cfg.projector_type} projector: shape={labels.shape}, dtype={labels.dtype}")
                
            # Add text_mask parameter if it exists in the signature
            if "text_mask" in forward_params:
                projector_args["text_mask"] = text_mask
                
            # For QFormer-specific implementations that use text_embeddings instead of text_tokens
            if "text_embeddings" in forward_params and "text_tokens" not in forward_params:
                projector_args["text_embeddings"] = labels_embedding_matched
                # Log for debugging - remove later
                logger.info(f"Passing text_embeddings to {self.cfg.projector_type} projector: shape={labels_embedding_matched.shape}, dtype={labels_embedding_matched.dtype}")

            # Special handling for visual_speech_text_qformer
            if self.cfg.projector_type.lower() == "visual_speech_text_qformer":
                logger.info(f"Using visual_speech_text_qformer with text input")
                # Ensure we're explicitly handling text tokens
                if "text_tokens" not in projector_args:
                    projector_args["text_tokens"] = labels
                if "text_mask" not in projector_args:
                    # Create attention mask (1 for real tokens, 0 for padding)
                    projector_args["text_mask"] = (labels != 0)
                
                # Log some diagnostic information about CTC targets
                if self.batch_counter % 20 == 0:
                    non_zero_targets = (labels != 0).sum().item()
                    total_targets = labels.size(0)
                    logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                    logger.info(f"CTC input_lengths: min={labels.min().item()}, max={labels.max().item()}")
                    logger.info(f"CTC target_lengths: min={labels.min().item()}, max={labels.max().item()}")
                    
                    # Check if we have invalid CTC constraints (input shorter than target)
                    invalid_lens = (labels == 0).sum().item()
                    if invalid_lens > 0:
                        logger.warning(f"CTC constraint violation: {invalid_lens} sequences have input_length < target_length")

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
            # Choose the CTC feature source based on configuration
            if self.cfg.ctc_feature_source == "encoder":
                # Use raw encoder output for CTC
                ctc_features = raw_encoder_out.clone()
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_encoder.weight.dtype)
                
                # Apply dropout for regularization during training (if in training mode)
                if self.training:
                    ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                    
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
            else:
                # Use projector output for CTC
                ctc_features = output['encoder_out'].clone()
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_projector.weight.dtype)
                
                # Apply dropout for regularization during training (if in training mode)
                if self.training:
                    ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                    
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
            
            # Apply temperature scaling to make distributions less peaked (especially during inference)
            temperature = 1.5 if not self.training else 1.0
            ctc_logits = ctc_logits / temperature
            
            # Apply log softmax
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
            
            # Get input lengths from encoder output
            if 'padding_mask' in output and output['padding_mask'] is not None:
                input_lengths = (~output['padding_mask']).long().sum(-1)
            else:
                input_lengths = torch.full((ctc_features.size(0),), 
                                        ctc_features.size(1),
                                        device=ctc_features.device,
                                        dtype=torch.long)
            
            # Convert word-level targets to character-level tokens for CTC
            ctc_tokens, ctc_lengths = self.get_ctc_targets_from_llm_batch(kwargs)
            
            # Add diagnostic logs to help debug CTC issues
            if self.batch_counter % 20 == 0:
                non_zero_targets = (ctc_lengths > 0).sum().item()
                total_targets = ctc_lengths.size(0)
                logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                logger.info(f"CTC input_lengths: min={input_lengths.min().item()}, max={input_lengths.max().item()}")
                logger.info(f"CTC target_lengths: min={ctc_lengths.min().item()}, max={ctc_lengths.max().item()}")
            
            # CTC needs float32 for loss calculation
            log_probs_float = ctc_log_probs.float()
            
            # Transpose from [B, T, C] to [T, B, C] for CTC loss
            log_probs_float = log_probs_float.transpose(0, 1)
            
            # Ensure input_lengths don't exceed actual sequence length
            max_time = log_probs_float.size(0)
            input_lengths = torch.clamp(input_lengths, max=max_time)
            
            # For CTC to work, target sequences must be shorter than input sequences
            # Check and fix any violations
            invalid_idx = input_lengths <= ctc_lengths
            if invalid_idx.any():
                # Log this occurrence
                logger.warning(f"Found {invalid_idx.sum().item()} sequences where input_length <= target_length, fixing")
                
                # Fix by ensuring all input lengths are at least target_length + 1
                min_valid_length = ctc_lengths + 1
                input_lengths = torch.maximum(input_lengths, min_valid_length)
                
                # If this makes any input length exceed the max sequence length, we need to pad
                if (input_lengths > max_time).any():
                    logger.warning(f"Need to pad input sequence to accommodate target lengths")
                    # Calculate new max time needed
                    new_max_time = input_lengths.max().item()
                    
                    # Create padding for the log_probs tensor (initialize with log of a very small probability)
                    pad_size = new_max_time - max_time
                    # Fill with blank token distribution (most probability mass on blank)
                    blank_hot = torch.full((pad_size, log_probs_float.size(1), log_probs_float.size(2)), 
                                          -20.0, device=log_probs_float.device)
                    # Give the blank token a higher probability
                    blank_hot[:, :, self.ctc_blank_idx] = 0.0
                    
                    # Concatenate to the original tensor
                    log_probs_float = torch.cat([log_probs_float, blank_hot], dim=0)
                    
                    # Update max_time
                    max_time = new_max_time
            
            # Prepare targets for CTC loss - we need to flatten targets
            # Extract valid (non-padding) tokens from ctc_tokens
            flat_targets = []
            batch_sizes = []
            
            # Debug information
            logger.info(f"CTC target shape before processing: {ctc_tokens.shape}")
            logger.info(f"CTC target lengths: {ctc_lengths}")
            
            for b in range(ctc_tokens.size(0)):
                # Get valid tokens for this sequence
                valid_tokens = ctc_tokens[b, :ctc_lengths[b]]
                flat_targets.append(valid_tokens)
                batch_sizes.append(valid_tokens.size(0))
            
            # Flatten targets into a 1D tensor
            flat_targets = torch.cat(flat_targets)
            
            # Debug info for the processed targets
            logger.info(f"Flattened target shape: {flat_targets.shape}")
            logger.info(f"Input lengths: {input_lengths}")
            
            # Calculate CTC loss
            ctc_loss = F.ctc_loss(
                log_probs_float,
                flat_targets,
                input_lengths,
                ctc_lengths,
                blank=self.ctc_blank_idx,
                reduction="mean",
                zero_infinity=True,
            )
            
            # Combine losses with weighting
            total_loss = (1 - self.cfg.ctc_weight) * lm_loss + self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype)
            
            # Log losses periodically
            if self.batch_counter % 20 == 0:
                logger.info(f"Losses - LM: {lm_loss.item():.4f}, CTC: {ctc_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        # Return combined loss, logits, and labels as expected by the criterion
        return total_loss, llm_out.logits, llm_labels

    def get_ctc_emissions(self, output, feature_source="projector"):
        """Get CTC emissions from encoder or projector output."""
        if feature_source == "encoder":
            ctc_features = output['encoder_out'].clone()
            ctc_features_matched = ctc_features.to(dtype=self.ctc_head_encoder.weight.dtype)
            
            # Apply dropout for regularization during training (if in training mode)
            if self.training:
                ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                
            ctc_logits = self.ctc_head_encoder(ctc_features_matched)
        else:
            # For projector source, we need to apply the projector first
            # Get the projector output
            proj_output = self.avfeat_to_llm(output['encoder_out'])
            ctc_features_matched = proj_output.to(dtype=self.ctc_head_projector.weight.dtype)
            
            # Apply dropout for regularization during training (if in training mode)
            if self.training:
                ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                
            ctc_logits = self.ctc_head_projector(ctc_features_matched)
        
        # Apply temperature scaling to make distributions less peaked (especially during inference)
        temperature = 1.5 if not self.training else 1.0
        ctc_logits = ctc_logits / temperature
        
        # Apply log softmax
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        
        return ctc_log_probs

    @torch.no_grad()
    def generate(self,
                num_beams=20,
                max_length=30,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=0.0,
                **kwargs,
                ):
        output = self.encoder(**kwargs)
        
        # Get CTC emissions if enabled
        if self.cfg.use_ctc:
            ctc_log_probs = self.get_ctc_emissions(output, self.cfg.ctc_feature_source)
            # Decode CTC
            ctc_decoded = self.decode_ctc(ctc_log_probs.transpose(0, 1))
            logger.info(f"CTC decoded text: {ctc_decoded}")
        
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
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (text_tokens != 0)
            
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
        outputs = self.decoder.generate(
            inputs_embeds=llm_input,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            length_penalty=length_penalty,
        )

        return outputs

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
        """Upgrade old state dicts to work with newer code."""
        super().upgrade_state_dict_named(state_dict, name)
        
        # Handle any CTC-specific state dict upgrades here
        if name == "":
            # Check if this is an old checkpoint without CTC heads
            if 'ctc_head_encoder' not in state_dict and 'ctc_head_projector' not in state_dict:
                logger.info("Loading checkpoint without CTC heads - initializing new CTC heads")
                # The CTC heads will be initialized with default weights
                return state_dict
            
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
        """Save all model state including encoder, decoder, projectors, and CTC components."""
        # Get the full state dict from parent
        state = super().state_dict()
        
        # Log what we're saving for debugging
        logger.info("Saving model state with keys: " + ", ".join(state.keys()))
        
        return state

    def get_ctc_targets_from_llm_batch(self, batch):
        """
        Extract and prepare CTC targets from a batch containing LLM targets.
        
        Args:
            batch: Dictionary containing 'target_list' from the LLM
            
        Returns:
            ctc_tokens: tensor of shape [batch_size, max_seq_len] with character indices
            ctc_lengths: tensor of shape [batch_size] with lengths of each sequence
        """
        # Get the device
        device = batch['target_list'].device
        
        # Get raw text sequences from the batch
        text_batch = []
        
        # Enhanced logging
        logger.info(f"===== CTC Target Text Extraction =====")
        logger.info(f"Target list shape: {batch['target_list'].shape}")
        
        for idx, seq in enumerate(batch['target_list']):
            # Get non-padding indices
            valid_indices = seq[seq != 0].tolist()
            
            # Log raw indices
            logger.info(f"Sequence {idx} valid indices: {valid_indices[:20]}..." if len(valid_indices) > 20 else f"Sequence {idx} valid indices: {valid_indices}")
            
            # Create a deterministic and unique string for CTC supervision
            # Don't rely on LLM tokenizer - use a simpler approach
            
            # Use a simple hash of the token IDs to generate unique string
            token_repr = []
            for token_id in valid_indices:
                # Convert token ID to ASCII range (using modulo to keep in printable range)
                char_code = (token_id % 26) + 97  # 'a' to 'z'
                token_repr.append(chr(char_code))
            
            # Join the characters to form a string
            text = ''.join(token_repr)
            
            # Ensure minimum length (add padding if needed)
            if len(text) < 5:
                text += 'x' * (5 - len(text))
                
            # Log extracted text
            logger.info(f"Sequence {idx} text: '{text}' (length: {len(text)})")
            
            text_batch.append(text)
        
        # Convert text batch to CTC tokens
        ctc_tokens, ctc_lengths = self.text_to_ctc_tokens(text_batch, device)
        
        # Log token information
        logger.info(f"CTC tokens shape: {ctc_tokens.shape}")
        for idx, length in enumerate(ctc_lengths):
            logger.info(f"Sequence {idx} token length: {length}")
            if length > 0:
                logger.info(f"Sequence {idx} first few tokens: {ctc_tokens[idx, :min(5, length)].tolist()}")
        
        logger.info(f"================================")
        
        return ctc_tokens, ctc_lengths

    def text_to_ctc_tokens(self, text_batch, device):
        """
        Convert a batch of text strings to character-level CTC tokens.
        
        Args:
            text_batch: List of text strings
            device: Device to create tensors on
            
        Returns:
            ctc_tokens: tensor of shape [batch_size, max_seq_len] with character indices
            ctc_lengths: tensor of shape [batch_size] with lengths of each sequence
        """
        # For each item in batch, convert to character sequence
        batch_tokens = []
        batch_lengths = []
        
        for text in text_batch:
            # Convert text to lowercase and strip whitespace
            text = text.lower().strip()
            
            # Ensure text is not empty, add a space if needed
            if len(text) == 0:
                text = " "
                
            # Convert to character indices
            char_indices = [self.ctc_token_to_idx.get(c, self.ctc_token_to_idx[' ']) 
                          for c in text]
            
            # Store in batch
            batch_tokens.append(torch.tensor(char_indices, device=device))
            batch_lengths.append(len(char_indices))
        
        # Pad sequences to max length
        max_len = max(batch_lengths)
        padded_tokens = []
        
        for tokens, length in zip(batch_tokens, batch_lengths):
            # Pad with CTC pad token
            padding = torch.full((max_len - length,), self.ctc_pad_idx, 
                               device=device, dtype=torch.long)
            padded_tokens.append(torch.cat([tokens, padding]))
        
        # Stack into batch tensor
        ctc_tokens = torch.stack(padded_tokens)
        ctc_lengths = torch.tensor(batch_lengths, device=device)
        
        return ctc_tokens, ctc_lengths

    def decode_ctc(self, log_probs, input_lengths=None, beam_size=5):
        """Decode CTC log probabilities to text."""
        # Move to CPU for decoding
        log_probs = log_probs.cpu()
        if input_lengths is not None:
            input_lengths = input_lengths.cpu()
        
        decoded_seqs = []
        batch_size = log_probs.size(1)
        
        for b in range(batch_size):
            # Get sequence up to input length if provided
            if input_lengths is not None:
                seq_log_probs = log_probs[:input_lengths[b], b]
            else:
                seq_log_probs = log_probs[:, b]
            
            # Collapse repeated tokens and remove blanks
            # First get greedy predictions for each timestep
            pred_tokens = seq_log_probs.argmax(dim=-1).tolist()
            
            # Apply CTC decoding rules: collapse repeats and remove blanks
            decoded_tokens = []
            prev_token = -1  # Different from any token
            for token in pred_tokens:
                # Skip blanks
                if token == self.ctc_blank_idx:
                    continue
                # Skip repeats (unless separated by blank, which we've removed)
                if token != prev_token:
                    decoded_tokens.append(token)
                    prev_token = token
            
            # If we have no tokens after removing blanks, check if we have any non-blank predictions
            if len(decoded_tokens) == 0:
                # Find positions of non-blank predictions with highest probability
                for t in range(seq_log_probs.size(0)):
                    probs = torch.exp(seq_log_probs[t])
                    max_val, max_idx = probs.max(0)
                    # If there's a reasonable non-blank prediction, use it
                    if max_idx != self.ctc_blank_idx and max_val > 0.1:
                        decoded_tokens.append(max_idx.item())
            
            # If still empty, use a placeholder
            if len(decoded_tokens) == 0:
                text = " "  # Empty space instead of "blank" to be less intrusive
            else:
                # Convert tokens to text
                text = ''.join([self.ctc_vocab[t] for t in decoded_tokens])
            
            decoded_seqs.append(text)
        
        return decoded_seqs


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
        
        # Set up CTC character vocabulary (standard English characters + common symbols)
        # Index 0 = padding, Index 1 = blank token (CTC standard), rest = characters
        self.ctc_vocab = ['<pad>', '<blank>'] + list(' abcdefghijklmnopqrstuvwxyz0123456789.,?!-\'\":;()[]')
        self.ctc_vocab_size = len(self.ctc_vocab)
        self.ctc_token_to_idx = {token: idx for idx, token in enumerate(self.ctc_vocab)}
        self.ctc_blank_idx = 1  # Index of <blank> in self.ctc_vocab (for CTC loss)
        self.ctc_pad_idx = 0    # Index of <pad> in self.ctc_vocab (for padding)
        
        # Log CTC vocabulary details
        logger.info(f"Initialized character-level CTC vocabulary with {self.ctc_vocab_size} tokens")
        logger.info(f"CTC blank token is at index {self.ctc_blank_idx}, pad token at index {self.ctc_pad_idx}")
        
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
        
        # Log the determined dimensions
        logger.info(f"Encoder CTC head will use dimensions: {encoder_dim} → {self.ctc_vocab_size}")
        logger.info(f"Projector CTC head will use dimensions: {projector_out_dim} → {self.ctc_vocab_size}")
        
        # Add CTC heads for both feature sources using the character-level CTC vocab
        self.ctc_head_encoder = nn.Linear(encoder_dim, self.ctc_vocab_size)
        self.ctc_head_projector = nn.Linear(projector_out_dim, self.ctc_vocab_size)
        
        # Initialize the projection layers with a special bias for non-blank tokens
        # This helps avoid the model getting stuck predicting all blanks
        nn.init.xavier_normal_(self.ctc_head_encoder.weight, gain=0.1)
        nn.init.zeros_(self.ctc_head_encoder.bias)
        # Set a negative bias for the blank token to make other tokens more likely initially
        self.ctc_head_encoder.bias.data[self.ctc_blank_idx] = -2.0
        
        nn.init.xavier_normal_(self.ctc_head_projector.weight, gain=0.1)
        nn.init.zeros_(self.ctc_head_projector.bias)
        # Set a negative bias for the blank token to make other tokens more likely initially
        self.ctc_head_projector.bias.data[self.ctc_blank_idx] = -2.0
        
        # Log CTC configuration
        logger.info(f"==========================================================")
        logger.info(f"INITIALIZING MODEL WITH CTC: weight={cfg.ctc_weight}, blank={self.ctc_blank_idx}")
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
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'use_ctc'):
            self._use_ctc_in_forward = torch.is_grad_enabled() and self.cfg.use_ctc
        
        # First get encoder output
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        
        # Store raw encoder output for CTC loss if needed
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
                projector_args["text_tokens"] = labels  # Pass raw token IDs for BLIP2 and Comprehensive QFormer
                # Log for debugging - remove later
                logger.info(f"Passing text_tokens to {self.cfg.projector_type} projector: shape={labels.shape}, dtype={labels.dtype}")
                
            # Add text_mask parameter if it exists in the signature
            if "text_mask" in forward_params:
                projector_args["text_mask"] = text_mask
                
            # For QFormer-specific implementations that use text_embeddings instead of text_tokens
            if "text_embeddings" in forward_params and "text_tokens" not in forward_params:
                projector_args["text_embeddings"] = labels_embedding_matched
                # Log for debugging - remove later
                logger.info(f"Passing text_embeddings to {self.cfg.projector_type} projector: shape={labels_embedding_matched.shape}, dtype={labels_embedding_matched.dtype}")

            # Special handling for visual_speech_text_qformer
            if self.cfg.projector_type.lower() == "visual_speech_text_qformer":
                logger.info(f"Using visual_speech_text_qformer with text input")
                # Ensure we're explicitly handling text tokens
                if "text_tokens" not in projector_args:
                    projector_args["text_tokens"] = labels
                if "text_mask" not in projector_args:
                    # Create attention mask (1 for real tokens, 0 for padding)
                    projector_args["text_mask"] = (labels != 0)
                
                # Log some diagnostic information about CTC targets
                if self.batch_counter % 20 == 0:
                    non_zero_targets = (labels != 0).sum().item()
                    total_targets = labels.size(0)
                    logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                    logger.info(f"CTC input_lengths: min={labels.min().item()}, max={labels.max().item()}")
                    logger.info(f"CTC target_lengths: min={labels.min().item()}, max={labels.max().item()}")
                    
                    # Check if we have invalid CTC constraints (input shorter than target)
                    invalid_lens = (labels == 0).sum().item()
                    if invalid_lens > 0:
                        logger.warning(f"CTC constraint violation: {invalid_lens} sequences have input_length < target_length")

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
            # Choose the CTC feature source based on configuration
            if self.cfg.ctc_feature_source == "encoder":
                # Use raw encoder output for CTC
                ctc_features = raw_encoder_out.clone()
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_encoder.weight.dtype)
                
                # Apply dropout for regularization during training (if in training mode)
                if self.training:
                    ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                    
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
            else:
                # Use projector output for CTC
                ctc_features = output['encoder_out'].clone()
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_projector.weight.dtype)
                
                # Apply dropout for regularization during training (if in training mode)
                if self.training:
                    ctc_features_matched = F.dropout(ctc_features_matched, p=0.1, training=self.training)
                    
                ctc_logits = self.ctc_head_projector(ctc_features_matched)
            
            # Apply temperature scaling to make distributions less peaked (especially during inference)
            temperature = 1.5 if not self.training else 1.0
            ctc_logits = ctc_logits / temperature
            
            # Apply log softmax
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
            
            # Get input lengths from encoder output
            if 'padding_mask' in output and output['padding_mask'] is not None:
                input_lengths = (~output['padding_mask']).long().sum(-1)
            else:
                input_lengths = torch.full((ctc_features.size(0),), 
                                        ctc_features.size(1),
                                        device=ctc_features.device,
                                        dtype=torch.long)
            
            # Convert word-level targets to character-level tokens for CTC
            ctc_tokens, ctc_lengths = self.get_ctc_targets_from_llm_batch(kwargs)
            
            # Add diagnostic logs to help debug CTC issues
            if self.batch_counter % 20 == 0:
                non_zero_targets = (ctc_lengths > 0).sum().item()
                total_targets = ctc_lengths.size(0)
                logger.info(f"CTC target stats: {non_zero_targets}/{total_targets} sequences have non-zero length")
                logger.info(f"CTC input_lengths: min={input_lengths.min().item()}, max={input_lengths.max().item()}")
                logger.info(f"CTC target_lengths: min={ctc_lengths.min().item()}, max={ctc_lengths.max().item()}")
            
            # CTC needs float32 for loss calculation
            log_probs_float = ctc_log_probs.float()
            
            # Transpose from [B, T, C] to [T, B, C] for CTC loss
            log_probs_float = log_probs_float.transpose(0, 1)
            
            # Ensure input_lengths don't exceed actual sequence length
            max_time = log_probs_float.size(0)
            input_lengths = torch.clamp(input_lengths, max=max_time)
            
            # For CTC to work, target sequences must be shorter than input sequences
            # Check and fix any violations
            invalid_idx = input_lengths <= ctc_lengths
            if invalid_idx.any():
                # Log this occurrence
                logger.warning(f"Found {invalid_idx.sum().item()} sequences where input_length <= target_length, fixing")
                
                # Fix by ensuring all input lengths are at least target_length + 1
                min_valid_length = ctc_lengths + 1
                input_lengths = torch.maximum(input_lengths, min_valid_length)
                
                # If this makes any input length exceed the max sequence length, we need to pad
                if (input_lengths > max_time).any():
                    logger.warning(f"Need to pad input sequence to accommodate target lengths")
                    # Calculate new max time needed
                    new_max_time = input_lengths.max().item()
                    
                    # Create padding for the log_probs tensor (initialize with log of a very small probability)
                    pad_size = new_max_time - max_time
                    # Fill with blank token distribution (most probability mass on blank)
                    blank_hot = torch.full((pad_size, log_probs_float.size(1), log_probs_float.size(2)), 
                                          -20.0, device=log_probs_float.device)
                    # Give the blank token a higher probability
                    blank_hot[:, :, self.ctc_blank_idx] = 0.0
                    
                    # Concatenate to the original tensor
                    log_probs_float = torch.cat([log_probs_float, blank_hot], dim=0)
                    
                    # Update max_time
                    max_time = new_max_time
            
            # Prepare targets for CTC loss - we need to flatten targets
            # Extract valid (non-padding) tokens from ctc_tokens
            flat_targets = []
            batch_sizes = []
            
            # Debug information
            logger.info(f"CTC target shape before processing: {ctc_tokens.shape}")
            logger.info(f"CTC target lengths: {ctc_lengths}")
            
            for b in range(ctc_tokens.size(0)):
                # Get valid tokens for this sequence
                valid_tokens = ctc_tokens[b, :ctc_lengths[b]]
                flat_targets.append(valid_tokens)
                batch_sizes.append(valid_tokens.size(0))
            
            # Flatten targets into a 1D tensor
            flat_targets = torch.cat(flat_targets)
            
            # Debug info for the processed targets
            logger.info(f"Flattened target shape: {flat_targets.shape}")
            logger.info(f"Input lengths: {input_lengths}")
            
            # Calculate CTC loss
            ctc_loss = F.ctc_loss(
                log_probs_float,
                flat_targets,
                input_lengths,
                ctc_lengths,
                blank=self.ctc_blank_idx,
                reduction="mean",
                zero_infinity=True,
            )
            
            # Combine losses with weighting
            total_loss = (1 - self.cfg.ctc_weight) * lm_loss + self.cfg.ctc_weight * ctc_loss.to(lm_loss.dtype)
            
            # Log losses periodically
            if self.batch_counter % 20 == 0:
                logger.info(f"Losses - LM: {lm_loss.item():.4f}, CTC: {ctc_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        # Return combined loss, logits, and labels as expected by the criterion
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