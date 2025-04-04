#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified version of CTC methods for VSP-LLM model.
This removes the rescoring/hybrid functionality and focuses on basic CTC training and decoding.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def ctc_greedy_decode(model, features, feature_source="projector"):
    """
    Performs greedy CTC decoding on the given features.
    
    Args:
        model: The VSP-LLM model
        features: Tensor of shape [B, T, D] from encoder or projector
        feature_source: Whether to use "encoder" or "projector" CTC head
        
    Returns:
        List of decoded sequences as strings (if phonetic vocab) or token indices
    """
    # Skip if CTC not available
    if feature_source == "encoder" and model.ctc_head_encoder is None:
        logger.warning("Requested encoder CTC decoding but encoder CTC head is not available")
        return None
    elif feature_source == "projector" and model.ctc_head_projector is None:
        logger.warning("Requested projector CTC decoding but projector CTC head is not available")
        return None
        
    # Get logits from appropriate CTC head
    with torch.no_grad():
        if feature_source == "encoder":
            # Check if there's a dimension mismatch in CTC head
            expected_dim = next(model.ctc_head_encoder.parameters()).shape[1]
            actual_dim = features.shape[-1]
            
            if actual_dim != expected_dim:
                logger.warning(f"CTC dimension mismatch: model expects {expected_dim}, got {actual_dim}")
                logger.warning("This suggests the model was trained with different dimensions than being used for inference.")
                logger.warning("Please ensure the same model configuration is used for both training and inference.")
                return None
            
            features_matched = features.to(dtype=next(model.ctc_head_encoder.parameters()).dtype)
            logits = model.ctc_head_encoder(features_matched)
        else:
            # Check if there's a dimension mismatch in CTC head
            expected_dim = next(model.ctc_head_projector.parameters()).shape[1]
            actual_dim = features.shape[-1]
            
            if actual_dim != expected_dim:
                logger.warning(f"CTC dimension mismatch: model expects {expected_dim}, got {actual_dim}")
                logger.warning("This suggests the model was trained with different dimensions than being used for inference.")
                logger.warning("Please ensure the same model configuration is used for both training and inference.")
                return None
            
            features_matched = features.to(dtype=next(model.ctc_head_projector.parameters()).dtype)
            logits = model.ctc_head_projector(features_matched)
        
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
            if token != prev and token != model.cfg.ctc_blank_idx:
                decoded.append(token)
            prev = token
        
        # Convert to string if using phonetic vocabulary
        if hasattr(model, 'idx_to_phonetic'):
            decoded_str = ''.join([model.idx_to_phonetic.get(idx, '?') for idx in decoded])
            decoded_seqs.append(decoded_str)
        else:
            # Return indices
            decoded_seqs.append(decoded)
    
    return decoded_seqs

def ctc_beam_decode(model, features, beam_size=10, feature_source="projector"):
    """
    Performs beam search CTC decoding on the given features.
    
    Args:
        model: The VSP-LLM model
        features: Tensor of shape [B, T, D] from encoder or projector
        beam_size: Beam size for the search
        feature_source: Whether to use "encoder" or "projector" CTC head
        
    Returns:
        List of decoded sequences as strings (if phonetic vocab) or token indices
    """
    # Skip if CTC not available
    if feature_source == "encoder" and model.ctc_head_encoder is None:
        logger.warning("Requested encoder CTC beam decoding but encoder CTC head is not available")
        return None
    elif feature_source == "projector" and model.ctc_head_projector is None:
        logger.warning("Requested projector CTC beam decoding but projector CTC head is not available")
        return None
        
    # Get logits from appropriate CTC head
    with torch.no_grad():
        if feature_source == "encoder":
            # Check if there's a dimension mismatch in CTC head
            expected_dim = next(model.ctc_head_encoder.parameters()).shape[1]
            actual_dim = features.shape[-1]
            
            if actual_dim != expected_dim:
                logger.warning(f"CTC dimension mismatch: model expects {expected_dim}, got {actual_dim}")
                logger.warning("This suggests the model was trained with different dimensions than being used for inference.")
                logger.warning("Please ensure the same model configuration is used for both training and inference.")
                return None
            
            features_matched = features.to(dtype=next(model.ctc_head_encoder.parameters()).dtype)
            logits = model.ctc_head_encoder(features_matched)
        else:
            # Check if there's a dimension mismatch in CTC head
            expected_dim = next(model.ctc_head_projector.parameters()).shape[1]
            actual_dim = features.shape[-1]
            
            if actual_dim != expected_dim:
                logger.warning(f"CTC dimension mismatch: model expects {expected_dim}, got {actual_dim}")
                logger.warning("This suggests the model was trained with different dimensions than being used for inference.")
                logger.warning("Please ensure the same model configuration is used for both training and inference.")
                return None
            
            features_matched = features.to(dtype=next(model.ctc_head_projector.parameters()).dtype)
            logits = model.ctc_head_projector(features_matched)
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    
    # Simple beam search implementation
    batch_size = log_probs.size(0)
    vocab_size = log_probs.size(2)
    blank_idx = model.cfg.ctc_blank_idx
    
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
        if hasattr(model, 'idx_to_phonetic'):
            decoded_str = ''.join([model.idx_to_phonetic.get(idx, '?') for idx in best_beam])
            decoded_seqs.append(decoded_str)
        else:
            # Return indices
            decoded_seqs.append(best_beam)
    
    return decoded_seqs

def generate_with_ctc(model, output, reduced_enc_out, instruction, **kwargs):
    """
    Generate text using the LLM model with optional CTC decoding.
    
    Args:
        model: The VSP-LLM model
        output: Encoder output
        reduced_enc_out: Reduced encoder output after projector
        instruction: Instruction tokens
        **kwargs: Additional arguments for generation
    
    Returns:
        LLM outputs or dictionary of LLM and CTC outputs
    """
    # Get generation parameters with defaults
    num_beams = kwargs.get('num_beams', 1)
    top_p = kwargs.get('top_p', 0.9)
    min_length = kwargs.get('min_length', 1)
    length_penalty = kwargs.get('length_penalty', 1.0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.0)
    use_ctc_decoding = kwargs.get('use_ctc_decoding', False)
    return_ctc_outputs = kwargs.get('return_ctc_outputs', False)
    
    # Get instruction embedding with model structure awareness
    try:
        if hasattr(model.decoder, 'model') and hasattr(model.decoder.model, 'model'):
            instruction_embedding = model.decoder.model.model.embed_tokens(instruction)
        elif hasattr(model.decoder, 'model'):
            instruction_embedding = model.decoder.model.embed_tokens(instruction)
        else:
            instruction_embedding = model.decoder.embed_tokens(instruction)
    except AttributeError:
        # Fallback to get_input_embeddings method
        embedding_layer = model.decoder.get_input_embeddings()
        instruction_embedding = embedding_layer(instruction)
        
    llm_input = torch.cat((instruction_embedding, reduced_enc_out), dim=1) 

    # Use a consistent approach for all models
    model.decoder.config.use_cache = True
    
    # Get LLM outputs
    outputs = model.decoder.generate(
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
        num_return_sequences=1,  # Simplified to just return 1 sequence
    )
    
    # Store the original sequences for returning
    llm_output = outputs.sequences
    
    # If CTC decoding is not requested, return just the LLM output
    if not use_ctc_decoding or not hasattr(model, 'ctc_vocab_size'):
        return llm_output
    
    # Option to use CTC decoding
    ctc_decoded_outputs = None
    if model.cfg.ctc_feature_source == "encoder" and model.ctc_head_encoder is not None:
        # Use the encoder-based CTC head
        ctc_decoded_outputs = ctc_greedy_decode(model, output['encoder_out'].clone(), "encoder")
    elif model.ctc_head_projector is not None:
        # Use the projector-based CTC head
        ctc_decoded_outputs = ctc_greedy_decode(model, reduced_enc_out.clone(), "projector")
    else:
        # No CTC head available
        logger.warning("CTC decoding requested but appropriate CTC head is not available")
        return llm_output
    
    # Return both LLM and CTC outputs if requested
    if return_ctc_outputs:
        result = {
            'llm_output': llm_output,
            'ctc_output': ctc_decoded_outputs
        }
        return result
        
    # Otherwise just return LLM output
    return llm_output 