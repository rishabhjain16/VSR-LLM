# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
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

from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from omegaconf import II, MISSING
import contextlib
import inspect

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
        default=MISSING, metadata={"help": "path to hubert model"}
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
        default="linear", 
        metadata={"help": "Type of projector to use (linear, mlp, qformer, cross_attention, perceiver, adaptive_query, fusion_refinement)"}
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
    ctc_blank_idx: int = field(
        default=0, 
        metadata={"help": "Index to use for blank token in CTC"}
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
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
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
        
        # First get encoder output
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
    
        # Get the original size before applying projector
        orig_seq_len = output['encoder_out'].size(1)
        
        # Get transcript information for cross-modal alignment
        labels = kwargs['target_list'].clone()
        
        # Get labels embedding with model structure awareness
        try:
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                labels_embedding = self.decoder.model.model.embed_tokens(labels)
            elif hasattr(self.decoder, 'model'):
                labels_embedding = self.decoder.model.embed_tokens(labels)
            else:
                labels_embedding = self.decoder.embed_tokens(labels)
        except AttributeError:
            # Fallback to get_input_embeddings method
            embedding_layer = self.decoder.get_input_embeddings()
            labels_embedding = embedding_layer(labels)
        
        # Variable to store alignment loss
        alignment_loss = torch.tensor(0.0, device=output['encoder_out'].device)
        
        # Check if we're using a text-aware projector
        is_text_aware = 'text_aware' in self.cfg.projector_type or 'qformer' in self.cfg.projector_type.lower()
        
        # For all projectors, try to use text information if available
        has_text_support = hasattr(self.avfeat_to_llm, 'forward') and 'text_embeddings' in inspect.signature(self.avfeat_to_llm.forward).parameters
        
        if has_text_support:
            # Also pass transcript information for cross-modal alignment
            projector_output = self.avfeat_to_llm(
                output['encoder_out'], 
                text_tokens=labels,  # Pass transcript tokens for alignment
                text_embeddings=labels_embedding,  # Pass transcript embeddings for alignment
                return_alignment_loss=True
            )
            
            # Unpack if alignment loss was returned
            if isinstance(projector_output, tuple):
                output['encoder_out'], alignment_loss = projector_output
            else:
                output['encoder_out'] = projector_output
            
            # Log the use of transcript information
            if not self.logged_projector_shape:
                logger.info(f"Using transcript information for cross-modal alignment in {self.cfg.projector_type}")
        elif is_text_aware:
            # For text-aware projectors without text_embeddings support
            # Get text tokens from kwargs - these should always be present
            text_tokens = kwargs['source']['text']
            
            # Add debug info about text tokens
            if not self.logged_projector_shape:
                logger.info(f"Text tokens shape: {text_tokens.shape}, non-zero tokens: {(text_tokens != 0).sum().item()}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (text_tokens != 0)
            
            # Only log this once per run
            if not self.logged_projector_shape:
                logger.info(f"Using instruction tokens with shape {text_tokens.shape} for text-aware projector (consider upgrading to a projector that supports transcript alignment)")
            
            # Pass both video features and text to the projector
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'], text_tokens=text_tokens, text_mask=text_mask)
        else:
            # For non-text-aware projectors, just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        
        # Check if we're using a query-based projector that changes sequence length
        proj_seq_len = output['encoder_out'].size(1)
        
        # First check projector type name (more reliable)
        is_query_based = any(qp in self.cfg.projector_type.lower() for qp in [
            "qformer", "enhanced_qformer", "visual_speech_qformer", 
            "perceiver", "cross_attention", "adaptive_query",
            "blip2_qformer", "text_aware_qformer", "comprehensive_qformer",
            "text_aware_comprehensive_qformer", "visual_text_alignment", 
            "visual_speech_text_qformer", "ebranchformer_visual_speech"
        ])
        
        # Fallback to sequence length check if needed
        if not is_query_based:
            is_query_based = proj_seq_len != orig_seq_len
            
        # Handle different projector types
        if is_query_based:
            # For query-based projectors (QFormer, EnhancedQFormer, etc.) that return fixed number of tokens
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
        
        # Include alignment loss if it's available (for text-aware projectors)
        loss = llm_out.loss
        if has_text_support and alignment_loss.item() > 0:
            # Add alignment loss with a weight (e.g., 0.1)
            alignment_weight = 0.1
            total_loss = (1 - alignment_weight) * loss + alignment_weight * alignment_loss.to(loss.dtype)
            
            # Log losses occasionally
            if self.batch_counter % 20 == 0:
                loss_val = float(loss.detach().cpu().item())
                align_loss_val = float(alignment_loss.detach().cpu().item())
                logger.info(f"Batch {self.batch_counter}: LM loss: {loss_val:.4f}, "
                           f"Alignment loss: {align_loss_val:.4f}")
        else:
            total_loss = loss
        
        # Return loss, logits, and labels for all model types
        return total_loss, llm_out.logits, llm_labels


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
        
        # Check if we're using a text-aware projector
        is_text_aware = 'text_aware' in self.cfg.projector_type or 'qformer' in self.cfg.projector_type.lower()
        
        # For all projectors, try to use text information if available
        has_text_support = hasattr(self.avfeat_to_llm, 'forward') and 'text_embeddings' in inspect.signature(self.avfeat_to_llm.forward).parameters
        
        # In generate mode, we only have instruction tokens available
        # For projectors that support text embeddings or text-aware projectors
        if has_text_support or is_text_aware:
            # Get text tokens from kwargs (only instruction tokens are available during inference)
            text_tokens = kwargs['source']['text']
            
            # Add debug info about text tokens
            logger.info(f"Generate - Text tokens shape: {text_tokens.shape}, non-zero tokens: {(text_tokens != 0).sum().item()}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (text_tokens != 0)
            logger.info(f"Using instruction tokens in generate() mode (transcript tokens not available during inference)")
            
            # Pass both video features and text to the projector
            if has_text_support:
                # Try to pass the instruction tokens as both tokens and embeddings
                try:
                    if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                        text_embeddings = self.decoder.model.model.embed_tokens(text_tokens)
                    elif hasattr(self.decoder, 'model'):
                        text_embeddings = self.decoder.model.embed_tokens(text_tokens)
                    else:
                        text_embeddings = self.decoder.embed_tokens(text_tokens)
                except AttributeError:
                    # Fallback to get_input_embeddings method
                    embedding_layer = self.decoder.get_input_embeddings()
                    text_embeddings = embedding_layer(text_tokens)
                
                output['encoder_out'] = self.avfeat_to_llm(
                    output['encoder_out'], 
                    text_tokens=text_tokens,
                    text_embeddings=text_embeddings
                )
            else:
                # Basic text-aware projector
                output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'], text_tokens=text_tokens, text_mask=text_mask)
        else:
            # For non-text-aware projectors, just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
            
        # Handle different projector types
        proj_seq_len = output['encoder_out'].size(1)
        orig_seq_len = kwargs['source']['video'].size(1)  # Original sequence length
        
        # First check projector type name (more reliable)
        is_query_based = any(qp in self.cfg.projector_type.lower() for qp in [
            "qformer", "enhanced_qformer", "visual_speech_qformer", 
            "perceiver", "cross_attention", "adaptive_query",
            "blip2_qformer", "text_aware_qformer", "comprehensive_qformer",
            "text_aware_comprehensive_qformer", "visual_text_alignment", 
            "visual_speech_text_qformer", "ebranchformer_visual_speech"
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
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Track the number of updates for logging purposes"""
        super().set_num_updates(num_updates) 
        self.num_updates = num_updates
        self.batch_counter += 1
        return self.num_updates

    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if 'lora' in k or 'avfeat_to_llm' in k or 'encoder' in k}
        return state


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
        
        # Get vocabulary size for CTC output dimension
        vocab_size = None
        if hasattr(decoder, 'config') and hasattr(decoder.config, 'vocab_size'):
            vocab_size = decoder.config.vocab_size
        elif hasattr(decoder, 'vocab_size'):
            vocab_size = decoder.vocab_size
        
        if vocab_size is None:
            raise ValueError("Could not determine vocabulary size for CTC head output dimension.")
        
        # Log the determined dimensions
        logger.info(f"Encoder CTC head will use dimensions: {encoder_dim} → {vocab_size}")
        logger.info(f"Projector CTC head will use dimensions: {projector_out_dim} → {vocab_size}")
        
        # Add CTC heads for both feature sources
        self.ctc_head_encoder = nn.Linear(encoder_dim, vocab_size)
        self.ctc_head_projector = nn.Linear(projector_out_dim, vocab_size)
        
        # Initialize the projection layers
        nn.init.xavier_normal_(self.ctc_head_encoder.weight)
        nn.init.zeros_(self.ctc_head_encoder.bias)
        nn.init.xavier_normal_(self.ctc_head_projector.weight)
        nn.init.zeros_(self.ctc_head_projector.bias)
        
        # Log CTC configuration
        logger.info(f"==========================================================")
        logger.info(f"INITIALIZING MODEL WITH CTC: weight={cfg.ctc_weight}, blank={cfg.ctc_blank_idx}")
        logger.info(f"CTC feature source: {cfg.ctc_feature_source}")
        logger.info(f"==========================================================")
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        model = super().build_model(cfg, task)
        
        # Replace with our CTC-enabled version
        return VSP_LLM_With_CTC(model.encoder, model.decoder, cfg)
    
    def forward(self, **kwargs):
        # Increment batch counter for logging
        self.batch_counter += 1
        
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
        
        # Get labels embedding with model structure awareness
        try:
            if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
                labels_embedding = self.decoder.model.model.embed_tokens(labels)
            elif hasattr(self.decoder, 'model'):
                labels_embedding = self.decoder.model.embed_tokens(labels)
            else:
                labels_embedding = self.decoder.embed_tokens(labels)
        except AttributeError:
            # Fallback to get_input_embeddings method
            embedding_layer = self.decoder.get_input_embeddings()
            labels_embedding = embedding_layer(labels)
        
        # Variable to store alignment loss
        alignment_loss = torch.tensor(0.0, device=output['encoder_out'].device)
        
        # Check if we're using a text-aware projector
        is_text_aware = 'text_aware' in self.cfg.projector_type or 'qformer' in self.cfg.projector_type.lower()
        
        # For all projectors, try to use text information if available
        has_text_support = hasattr(self.avfeat_to_llm, 'forward') and 'text_embeddings' in inspect.signature(self.avfeat_to_llm.forward).parameters
        
        if has_text_support:
            # Also pass transcript information for cross-modal alignment
            projector_output = self.avfeat_to_llm(
                output['encoder_out'], 
                text_tokens=labels,  # Pass transcript tokens for alignment
                text_embeddings=labels_embedding,  # Pass transcript embeddings for alignment
                return_alignment_loss=True
            )
            
            # Unpack if alignment loss was returned
            if isinstance(projector_output, tuple):
                output['encoder_out'], alignment_loss = projector_output
            else:
                output['encoder_out'] = projector_output
            
            # Log the use of transcript information
            if not self.logged_projector_shape:
                logger.info(f"Using transcript information for cross-modal alignment in {self.cfg.projector_type}")
        elif is_text_aware:
            # For text-aware projectors without text_embeddings support
            # Get text tokens from kwargs - these should always be present
            text_tokens = kwargs['source']['text']
            
            # Add debug info about text tokens
            if not self.logged_projector_shape:
                logger.info(f"Text tokens shape: {text_tokens.shape}, non-zero tokens: {(text_tokens != 0).sum().item()}")
            
            # Create attention mask (1 for real tokens, 0 for padding)
            text_mask = (text_tokens != 0)
            
            # Only log this once per run
            if not self.logged_projector_shape:
                logger.info(f"Using instruction tokens with shape {text_tokens.shape} for text-aware projector (consider upgrading to a projector that supports transcript alignment)")
            
            # Pass both video features and text to the projector
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'], text_tokens=text_tokens, text_mask=text_mask)
        else:
            # For non-text-aware projectors, just pass the video features
            output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        
        # Check if we're using a query-based projector that changes sequence length
        proj_seq_len = output['encoder_out'].size(1)
        
        # First check projector type name (more reliable)
        is_query_based = any(qp in self.cfg.projector_type.lower() for qp in [
            "qformer", "enhanced_qformer", "visual_speech_qformer", 
            "perceiver", "cross_attention", "adaptive_query",
            "blip2_qformer", "text_aware_qformer", "comprehensive_qformer",
            "text_aware_comprehensive_qformer", "visual_text_alignment", 
            "visual_speech_text_qformer", "ebranchformer_visual_speech"
        ])
        
        # Fallback to sequence length check if needed
        if not is_query_based:
            is_query_based = proj_seq_len != orig_seq_len
            
        # Handle different projector types
        if is_query_based:
            # For query-based projectors (QFormer, EnhancedQFormer, etc.) that return fixed number of tokens
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
        if self.training and self.cfg.use_ctc:
            # Choose the CTC feature source based on configuration
            if self.cfg.ctc_feature_source == "encoder":
                # Use raw encoder output for CTC
                ctc_features = raw_encoder_out
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_encoder.weight.dtype)
                ctc_logits = self.ctc_head_encoder(ctc_features_matched)
                
                # Get input lengths from encoder output
                if 'padding_mask' in output and output['padding_mask'] is not None:
                    input_lengths = (~output['padding_mask']).long().sum(-1)
                else:
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
                ctc_features_matched = ctc_features.to(dtype=self.ctc_head_projector.weight.dtype)
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
            
            # Get target sequence (same text used for LLM loss)
            target = labels
            
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
                
                # Verify our blank index isn't causing issues with padding
                if self.cfg.ctc_blank_idx == 0:
                    logger.warning("CTC blank_idx is set to 0, which is also the padding token - this may cause issues")
            
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
                        
                        # Try with a fixed minimum loss to prevent zero gradient
                        if self.cfg.ctc_feature_source == "projector" and ctc_loss.item() < 1e-6:
                            # If using projector and getting zero loss, fall back to a nominal minimum loss
                            # to ensure some gradient signal reaches the CTC head
                            logger.warning("Applying minimum loss floor for numerical stability")
                            ctc_loss = torch.max(ctc_loss, torch.tensor(0.1, device=ctc_loss.device, dtype=ctc_loss.dtype))
                
                # Combine losses with weighting
                # Convert to same dtype for combination
                # Add alignment loss if available (for cross-modal learning)
                if has_text_support and alignment_loss.item() > 0:
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