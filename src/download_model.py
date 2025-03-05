#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, utils
from huggingface_hub.utils import HfHubHTTPError

# Configure logging to go to stderr instead of stdout
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,  # Send logs to stderr
)
logger = logging.getLogger("download_model")

# Disable logging from transformers (will still show errors)
utils.logging.set_verbosity_error()

def download_model(model_name, output_dir=None):
    """
    Downloads a model from Hugging Face if it doesn't exist locally.
    
    Args:
        model_name: The name or path of the model
        output_dir: The directory to save the model to. If None, uses model_name.
    
    Returns:
        Path to the downloaded model
    """
    # Check if model_name is a local path or a Hugging Face model ID
    if os.path.exists(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
        logger.info(f"Model already exists at {model_name}")
        return model_name
    
    # If not a local path, we need to download it
    try:
        # Determine the output directory
        if output_dir is None:
            # Extract just the model name from the HF path 
            model_basename = model_name.split('/')[-1]
            # If model_name doesn't contain '/', use it as is
            if model_basename == model_name:
                output_dir = model_name
            else:
                output_dir = model_basename
    
        output_dir = Path(output_dir)
        
        # Check if the model already exists at the output location
        if output_dir.exists() and os.path.isfile(os.path.join(output_dir, "config.json")):
            logger.info(f"Model already exists at {output_dir}")
            return str(output_dir)
        
        logger.info(f"Downloading model {model_name} to {output_dir}")
        
        # Create parent directories if they don't exist
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Download both the model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True,
                torch_dtype="auto",
            )
            
            # Save the model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Successfully downloaded model to {output_dir}")
            return str(output_dir)
            
        except HfHubHTTPError as e:
            if "401" in str(e):
                logger.error(f"Authentication error: You need to log in to access {model_name}")
                logger.error("Run 'huggingface-cli login' to authenticate with your HF account")
                logger.error("If this is a gated model, make sure you've requested and been granted access")
            elif "403" in str(e):
                logger.error(f"Access forbidden: You don't have permission to access {model_name}")
                logger.error("For gated models like Llama, you need to request access on the HF website")
            else:
                logger.error(f"HTTP error downloading model {model_name}: {e}")
            
            # Clean up empty directory
            if output_dir.exists():
                try:
                    output_dir.rmdir()
                except:
                    pass
            return None
        
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            # Clean up empty directory
            if output_dir.exists():
                try:
                    output_dir.rmdir()
                except:
                    pass
            return None
    
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face if it doesn't exist locally")
    parser.add_argument("model_name", help="The name or path of the model")
    parser.add_argument("--output-dir", help="The directory to save the model to")
    args = parser.parse_args()
    
    model_path = download_model(args.model_name, args.output_dir)
    if model_path:
        # Print ONLY the path to stdout without any extra logging
        print(model_path, flush=True)
        sys.exit(0)
    else:
        # If download failed, exit with non-zero status
        logger.error(f"Failed to download model {args.model_name}")
        sys.exit(1)

if __name__ == "__main__":
    main() 