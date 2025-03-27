#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
DATA_PATH=/home/rishabh/Desktop/Datasets/lrs3/433h_data    # path to train dataset dir

OUT_PATH=/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/trained/Llama2_proj_test_vision   # output path to save

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

# Set the path to your LLM model here
# You can use any HuggingFace compatible model like:
# - Llama models: Meta-Llama-3-8B, Llama-2-7b-hf, etc.
# - Mistral models: mistralai/Mistral-7B-v0.1
# - Other models: EleutherAI/gpt-j-6b, facebook/opt-6.7b, etc.

# IMPORTANT: For gated models like Llama-2 and Llama-3, you need to:
#  1. Create a Hugging Face account: https://huggingface.co/join
#  2. Request access to the model: https://huggingface.co/meta-llama/Llama-2-13b-hf
#  3. Log in with: huggingface-cli login
#
# If you don't have access, use an open model like:
# - "mistralai/Mistral-7B-v0.1"
# - "stabilityai/stablelm-3b-4e1t"
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_MODEL_ID="Llama-2-7b-hf"  # HuggingFace model ID
CHECKPOINT_DIR="${ROOT}/checkpoints"

# Check if the model exists locally, if not download it
MODEL_BASENAME=$(basename "$HF_MODEL_ID")
LLM_PATH="${CHECKPOINT_DIR}/${MODEL_BASENAME}"

# Create checkpoints directory if it doesn't exist
mkdir -p "${CHECKPOINT_DIR}"

# If LLM_PATH doesn't exist, download the model
if [ ! -d "${LLM_PATH}" ] || [ -z "$(ls -A ${LLM_PATH} 2>/dev/null)" ]; then
    echo "Model not found at ${LLM_PATH}, downloading from HuggingFace..."
    
    # Make download_model.py executable if it's not already
    chmod +x ${SRC}/download_model.py
    
    # Run the Python script to download the model and capture its output
    downloaded_path=$(python3 ${SRC}/download_model.py "$HF_MODEL_ID" --output-dir "${LLM_PATH}" 2>&1)
    
    # Check if the download was successful by looking for specific success message
    if echo "$downloaded_path" | grep -q "Successfully downloaded model"; then
        # Extract just the path from the output (last line)
        model_path=$(echo "$downloaded_path" | tail -n 1)
        LLM_PATH="${model_path}"
        echo "Model successfully downloaded to: ${LLM_PATH}"
    else
        echo "ERROR: Failed to download model. Output was:"
        echo "$downloaded_path"
        echo ""
        echo "This might be because:"
        echo "1. You need to authenticate with Hugging Face (run 'huggingface-cli login')"
        echo "2. You don't have access to this gated model"
        echo "3. There's a network or permission issue"
        echo ""
        echo "Try using an open model instead by changing HF_MODEL_ID to:"
        echo "- 'mistralai/Mistral-7B-v0.1'"
        echo "- 'stabilityai/stablelm-3b-4e1t'"
        echo "- 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'"
        exit 1
    fi
    
    # Verify the model files exist
    if [ ! -d "${LLM_PATH}" ] || [ ! -f "${LLM_PATH}/config.json" ]; then
        echo "ERROR: Model download seemed successful but files are missing."
        echo "Expected to find model files in: ${LLM_PATH}"
        exit 1
    fi
else
    echo "Using existing model at: ${LLM_PATH}"
fi

PRETRAINED_MODEL_PATH=${ROOT}/checkpoints/large_vox_iter5.pt   # path to pretrained avhubert large_lrs3_iter5

# Note: The code has been updated to automatically:
#  - Detect model architecture and adapt LoRA parameters accordingly
#  - Handle different model hidden sizes (encoder dimensions)
#  - Configure tokenizers appropriately for different models
#  - Apply model-specific prompt templates for optimal performance
# You should not need to manually modify the code for different model architectures

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

# CONFIGURATION:
# -------------------------------------------------------------
# Edit these variables directly to change training configuration
# -------------------------------------------------------------

# Which projector to use (linear, mlp, qformer, visual_speech_qformer, ebranchformer_cluster, etc.)
PROJECTOR_TYPE="linear"

# Whether to use attention-weighted cluster aggregation (true) or simple mean (false)
# Only applies to non-query-based projectors like linear, mlp, ebranchformer_cluster
USE_ATTENTION_CLUSTER=true

# -------------------------------------------------------------
# End of configuration section
# -------------------------------------------------------------

echo "Training with:"
echo "- Projector type: $PROJECTOR_TYPE"
if [[ "$PROJECTOR_TYPE" != *"qformer"* ]] && [[ "$PROJECTOR_TYPE" != *"perceiver"* ]] && [[ "$PROJECTOR_TYPE" != *"fusion"* ]]; then
    echo "- Using attention-weighted clustering: $USE_ATTENTION_CLUSTER"
else
    echo "- Query-based projector (clustering method doesn't apply)"
fi

fairseq-hydra-train \
    --config-dir ${SRC}/conf \
    --config-name vsp-llm-433h-freeze \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        +model.projector_type=${PROJECTOR_TYPE} \
        +model.use_attention_cluster=${USE_ATTENTION_CLUSTER} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 
