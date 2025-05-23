#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=en    # language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)

# set paths
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
MODEL_SRC=${ROOT}/src

# Set the path to your Llama 3 model here
#LLM_PATH=${ROOT}/checkpoints/vicuna-7b-v1.5    # path to vicuna checkpoint
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf
#LLM_PATH=${ROOT}/checkpoints/Meta-Llama-3-8B    # path to Llama 3 model

DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs3/433h_data     # path to test dataset dir
#DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs2_clean/data_lrs2
#DATA_ROOT=/home/rishabh/Desktop/Datasets/test_lrs2/lrs2/lrs2_video_seg16s/data_lrs2
#DATA_ROOT=/home/rishabh/Desktop/Datasets/WildVSR/test_data

# Note: For different models, you may need to manually modify:
# - src/vsp_llm.py: target_modules, lora_r, lora_alpha for the specific model architecture
# - Default values are set for Llama models and should work with Llama 3
#DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs2_clean/data_lrs2
#DATA_ROOT=/home/rishabh/Desktop/Datasets/test_lrs2/lrs2/lrs2_video_seg16s/data_lrs2
#DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs2/auto/lrs2/lrs2_video_seg24s/data_lrs2
#DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs2_rf/lrs2/lrs2_video_seg16s/data_lrs2

MODEL_PATH=/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/llama2_comprehensive_qformer/checkpoints/checkpoint_best.pt   # path to trained model with Llama 3
#MODEL_PATH=${ROOT}/checkpoints/OG/checkpoint_finetune.pt
#MODEL_PATH=${ROOT}/checkpoints/checkpoint_finetune.pt  # path to trained model
#MODEL_PATH=/home/rijain@ad.mee.tcd.ie/Experiments/vsr-llm/checkpoints/checkpoint_finetune.pt
##MODEL_PATH=/home/rjain/experiments/VSR-LLM/checkpoints/checkpoint_finetune.pt
#MODEL_PATH=/home/rjain/experiments/VSR-LLM/checkpoints/checkpoint_vicuna_lrs3_50k.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_l3Prompt/checkpoints/checkpoint_best.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_UnordodoxPromptNew_bs4/checkpoints/checkpoint_best.pt


OUT_PATH=${ROOT}/checkpoints/decode/decode_visual_only_qformer   # output path to save results
#OUT_PATH=${ROOT}/checkpoints/decode/decode_test_mytrained_lora_32
#OUT_PATH=${ROOT}/checkpoints/decode/decode_test

# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="vst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    USE_BLEU=true
    DATA_PATH=${DATA_ROOT}/${TASK}/${SRC}/${TGT}

else
    TASK="vsr"
    TGT=${LANG}
    USE_BLEU=false
    DATA_PATH=${DATA_ROOT}
fi

# start decoding
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python -B ${MODEL_SRC}/vsp_llm_decode.py \
    --config-dir ${MODEL_SRC}/conf \
    --config-name s2s_decode \
        common.user_dir=${MODEL_SRC} \
        dataset.gen_subset=test \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        generation.beam=20 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG} \
