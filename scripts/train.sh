#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
DATA_PATH=/home/rishabh/Desktop/Datasets/lrs_combined   # path to train dataset dir

OUT_PATH=/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/trained/Non_CTC_L2_visual_only_qformer   # output path to save

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

HF_MODEL_ID="Llama-2-7b-hf"  # HuggingFace model ID
CHECKPOINT_DIR="${ROOT}/checkpoints"

# Check if the model exists locally, if not download it
MODEL_BASENAME=$(basename "$HF_MODEL_ID")
LLM_PATH="${CHECKPOINT_DIR}/${MODEL_BASENAME}"

# Create checkpoints directory if it doesn't exist
mkdir -p "${CHECKPOINT_DIR}"


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
PROJECTOR_TYPE="visual_only_qformer"

# CTC configuration
USE_CTC="false"  # Set to "true" to enable CTC loss
CTC_WEIGHT="0.3"  # Weight for CTC loss (0.3 means 30% CTC, 70% LM)
CTC_FEATURE_SOURCE="projector"  # Source of features for CTC: "encoder" or "projector"



# Check if CTC should be enabled
if [ "$USE_CTC" = "true" ]; then
    MODEL_TYPE="vsp_llm_ctc"
    echo "- Using CTC loss with weight: $CTC_WEIGHT"
    echo "- CTC feature source: $CTC_FEATURE_SOURCE"
else
    MODEL_TYPE="vsp_llm"
    echo "- Not using CTC loss"
fi

# Check if this is a query-based projector
if [[ "$PROJECTOR_TYPE" != *"qformer"* ]] && [[ "$PROJECTOR_TYPE" != *"cross_attention"* ]]; then
    echo "- Using clustering Approach"
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
        +task.projector_type=${PROJECTOR_TYPE} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        +model.projector_type=${PROJECTOR_TYPE} \
        +model.use_ctc=${USE_CTC} \
        +model.ctc_weight=${CTC_WEIGHT} \
        +model.ctc_feature_source=${CTC_FEATURE_SOURCE} \
        +override.disable_text_conditioning=true \
        model._name=${MODEL_TYPE} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 
