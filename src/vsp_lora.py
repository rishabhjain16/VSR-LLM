"""
Module for identifying optimal LoRA target modules for various LLM architectures.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel

logger = logging.getLogger(__name__)

# Dictionary of recommended target modules for known model architectures
MODEL_TYPE_TO_TARGET_MODULES = {
    'llama': ["q_proj", "v_proj", "k_proj", "o_proj"],
    'mistral': ["q_proj", "v_proj", "k_proj", "o_proj"],
    'phi': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    'phi3': ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    'falcon': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    'qwen': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    'qwen2_5_vl': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    'vision2seq': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    'gpt_neox': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    'gptj': ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"],
    'mpt': ["Wqkv", "out_proj", "fc1", "fc2"],
    'gemma': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    'default': ["q_proj", "v_proj", "k_proj", "o_proj"]
}

# Fixed LoRA parameters to use for all models
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'bias': "none",
    'task_type': "CAUSAL_LM"
}

def get_target_modules(model_or_type: Union[str, PreTrainedModel], verbose: bool = True) -> List[str]:
    """
    Identify the best target modules for LoRA fine-tuning.
    
    Args:
        model_or_type: Either the model itself, or the model type as a string 
                      (e.g., "llama", "mistral", etc.)
        verbose: Whether to print information about detected modules
        
    Returns:
        List of module names recommended for LoRA fine-tuning
    """
    # If input is already a string (model type), use it directly
    if isinstance(model_or_type, str):
        model_type = model_or_type.lower()
        if verbose:
            logger.info(f"Using provided model type: {model_type}")
        
        # Check if we have predefined target modules for this model type
        if model_type in MODEL_TYPE_TO_TARGET_MODULES:
            target_modules = MODEL_TYPE_TO_TARGET_MODULES[model_type]
            if verbose:
                logger.info(f"Using predefined target modules for {model_type}: {target_modules}")
            return target_modules
        else:
            # If the model type is unknown, use a simple default
            if verbose:
                logger.info(f"Unknown model type: {model_type}, using default target modules")
            return MODEL_TYPE_TO_TARGET_MODULES["default"]
    
    # If input is a model, detect the type and find target modules
    try:
        # Get model type
        if hasattr(model_or_type, 'config') and hasattr(model_or_type.config, 'model_type'):
            model_type = model_or_type.config.model_type.lower()
            
            # Special case for vision-language models
            if model_type == "vision2seq" or "qwen" in model_type and "vl" in model_type:
                if verbose:
                    logger.info(f"Detected vision-language model type: {model_type}")
                # Use vision2seq modules or qwen_vl specific ones if available
                if "qwen" in model_type and "vl" in model_type:
                    return MODEL_TYPE_TO_TARGET_MODULES.get("qwen2_5_vl", MODEL_TYPE_TO_TARGET_MODULES["vision2seq"])
                else:
                    return MODEL_TYPE_TO_TARGET_MODULES["vision2seq"]
            
            # Check if we have predefined target modules for this model type
            if model_type in MODEL_TYPE_TO_TARGET_MODULES:
                target_modules = MODEL_TYPE_TO_TARGET_MODULES[model_type]
                if verbose:
                    logger.info(f"Using predefined target modules for detected model type {model_type}: {target_modules}")
                return target_modules
            else:
                # For unknown models, analyze structure to find common modules
                target_modules = _analyze_model_structure(model_or_type)
                if verbose:
                    logger.info(f"Selected target modules for detected model type {model_type}: {target_modules}")
                return target_modules
        else:
            # If model type can't be detected, analyze structure
            target_modules = _analyze_model_structure(model_or_type)
            if verbose:
                logger.info(f"Using detected target modules: {target_modules}")
            return target_modules
    except Exception as e:
        logger.warning(f"Error detecting model structure: {e}. Using default target modules.")
        return MODEL_TYPE_TO_TARGET_MODULES["default"]

def _analyze_model_structure(model: PreTrainedModel) -> List[str]:
    """
    Analyze a model's structure to identify potential LoRA target modules.
    
    Args:
        model: The pre-trained model to analyze
        
    Returns:
        List of recommended module names for LoRA fine-tuning
    """
    # Patterns for identifying attention and MLP layers
    attention_patterns = [
        r'.*q_proj$', r'.*v_proj$', r'.*k_proj$', r'.*o_proj$',
        r'.*query$', r'.*value$', r'.*key$', r'.*attention', 
        r'.*qkv_proj$',
        r'.*gate_proj$', r'.*up_proj$', r'.*down_proj$',
        r'.*dense_h_to_4h$', r'.*dense_4h_to_h$',
        r'.*out_proj$'
    ]
    
    # Count occurrences of different module names
    module_counts = Counter()
    
    # Find all linear layers and their names
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or "Linear" in str(type(module)):
            # Extract the leaf module name (last part of the name)
            module_type = name.split('.')[-1]
            
            # Check if this matches our attention patterns
            if any(re.match(pattern, module_type) for pattern in attention_patterns):
                module_counts[module_type] += 1
    
    # If no modules found with our patterns, return default
    if not module_counts:
        return MODEL_TYPE_TO_TARGET_MODULES["default"]
    
    # Get most common modules (those that appear multiple times)
    common_modules = [name for name, count in module_counts.most_common() if count > 1]
    
    # If we found common modules, use them (up to 8 max)
    if common_modules:
        return common_modules[:8]
    else:
        return MODEL_TYPE_TO_TARGET_MODULES["default"]
