#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Set the projector type directly in this file - change this line to switch projector
PROJECTOR_TYPE="mlp"  # Options: linear, mlp, transformer, convolutional, cross_modal_attention, gated

# set variables
DATA_PATH=/home/rishabh/Desktop/Datasets/lrs3/433h_data    # path to train dataset dir
OUT_PATH=/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/trained/output_AV_VOX_433_with_${PROJECTOR_TYPE}
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf
PRETRAINED_MODEL_PATH=${ROOT}/checkpoints/large_vox_iter5.pt   # path to pretrained avhubert large_lrs3_iter5

# Remove all projector-specific arguments setup

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

echo "Training with $PROJECTOR_TYPE projector"

# Add additional training stabilization parameters
fairseq-hydra-train \
    --config-dir ${SRC}/conf \
    --config-name vsp-llm-433h-freeze \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        model.projector_type=${PROJECTOR_TYPE} \
        optimization.lr=[0.0002] \
        optimization.clip_norm=1.0 \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1
