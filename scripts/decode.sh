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

#LLM_PATH=${ROOT}/checkpoints/vicuna-7b-v1.5    # path to llama checkpoint
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf  
#DATA_ROOT=/home/rjain/data/lrs3/433h_data/    # path to test dataset dir
#DATA_ROOT=/home/rishabh/Desktop/Dataset/lrs2likelrs3/lrs2_video_seg16s/data_lrs2
DATA_ROOT=/home/rishabh/Desktop/Datasets/lrs3/433h_data

MODEL_PATH=${ROOT}/checkpoints/OG/checkpoint_finetune.pt  
#MODEL_PATH=${ROOT}/checkpoints/checkpoint_finetune.pt  # path to trained model
#MODEL_PATH=/home/rijain@ad.mee.tcd.ie/Experiments/vsr-llm/checkpoints/checkpoint_finetune.pt
##MODEL_PATH=/home/rjain/experiments/VSR-LLM/checkpoints/checkpoint_finetune.pt
#MODEL_PATH=/home/rjain/experiments/VSR-LLM/checkpoints/checkpoint_vicuna_lrs3_50k.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_l3Prompt/checkpoints/checkpoint_best.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_UnordodoxPromptNew_bs4/checkpoints/checkpoint_best.pt


#OUT_PATH=${ROOT}/decode/decode_vsp_433_freeze_my_training_llama_3.2_1B_T3_l3_UnordPrompt_infer_normal_prompt_9feb_lrs3_ck_8_70k_debug_lrs3    # output path to save
#OUT_PATH=${ROOT}/checkpoints/decode/decode_test_mytrained_lora_32
OUT_PATH=${ROOT}/checkpoints/decode/decode_test

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
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG}