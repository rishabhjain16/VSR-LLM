#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
DATA_PATH=/home/rjain/data/lrs3    # path to train dataset dir

OUT_PATH=/home/rjain/Experiments/VSR-LLM/checkpoints/training/output_AV_VOX_433_with_Vicuna1.5_lrs2_50kupdates_8bit    # output path to save

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

LLM_PATH=${ROOT}/checkpoints/vicuna-7b-v1.5
PRETRAINED_MODEL_PATH=${ROOT}/checkpoints/large_vox_iter5.pt   # path to pretrained avhubert large_lrs3_iter5

export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
fairseq-hydra-train \
    --config-dir ${SRC}/conf \
    --config-name vsp-llm-433h-freeze \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 
