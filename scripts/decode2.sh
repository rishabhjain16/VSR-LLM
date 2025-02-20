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
LLM_PATH=${ROOT}/checkpoints/vicuna-7b-v1.5    # path to llama checkpoint
DATA_ROOT=/home/rjain/data/lrs3    # path to test dataset dir
#DATA_ROOT=/home/rishabh/Desktop/Dataset/WildVSR

#MODEL_PATH=${ROOT}/checkpoints/checkpoint_finetune.pt  # path to trained model

MODEL_PATH=/home/rjain/Experiments/VSR-LLM/checkpoints/trained/output_AV_VOX_433_with_Vicuna1.5_lrs3_50kupdates/checkpoints/checkpoint_best.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_l3Prompt/checkpoints/checkpoint_best.pt
#MODEL_PATH=/home/rishabh/Desktop/Experiments/VSP-LLM/output_ckps/output_AV_VOX_433_with_Llama-3.2-1B_Training2_lrs3_70000updates_UnordodoxPromptNew_bs4/checkpoints/checkpoint_best.pt
#OUT_PATH=${ROOT}/decode/decode_vsp_433_freeze_my_training_llama_3.2_1B_T3_l3_UnordPrompt_infer_normal_prompt_9feb_lrs3_ck_8_70k_debug_lrs3    # output path to save
OUT_PATH=${ROOT}/checkpoints/decode/vsp_433_output_AV_VOX_433_with_Vicuna1.5_lrs3_on_lrs3

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

# export CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=0 python -B ${MODEL_SRC}/vsp_llm_decode.py \
#   --config-dir ${MODEL_SRC}/conf \
#   --config-name s2s_decode \
#   common.user_dir=${MODEL_SRC} \
#   dataset.gen_subset=test \
#   override.data=${DATA_PATH} \
#   override.label_dir=${DATA_PATH} \
#   generation.beam=20 \
#   generation.lenpen=0 \
#   dataset.max_tokens=3000 \
#   override.eval_bleu=${USE_BLEU} \
#   override.llm_ckpt_path=${LLM_PATH} \
#   common_eval.path=${MODEL_PATH} \
#   common_eval.results_path=${OUT_PATH}/${TASK}/${LANG} \

# ...existing code...

# export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
# CUDA_VISIBLE_DEVICES=0 python -B ${MODEL_SRC}/vsp_llm_decode.py \
#     --config-dir ${MODEL_SRC}/conf \
#     --config-name s2s_decode \
#         common.user_dir=${MODEL_SRC} \
#         dataset.gen_subset=test \
#         override.data=${DATA_PATH} \
#         override.label_dir=${DATA_PATH} \
#         override.llm_ckpt_path=${LLM_PATH} \
#         generation.beam=20 \
#         generation.max_len_b=50 \
#         generation.min_len=1 \
#         generation.lenpen=0.0 \
#         +generation.no_repeat_ngram_size=3 \
#         +generation.sampling=true \
#         +generation.sampling_topp=0.9 \
#         dataset.max_tokens=3000 \
#         common_eval.path=${MODEL_PATH} \
#         common_eval.quiet=false \
#         common_eval.results_path=${OUT_PATH} \
#         +generation.skip_invalid_size_inputs=true

# export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
# CUDA_VISIBLE_DEVICES=0 python -B ${MODEL_SRC}/vsp_llm_decode.py \
#     --config-dir ${MODEL_SRC}/conf \
#     --config-name s2s_decode \
#         common.user_dir=${MODEL_SRC} \
#         dataset.gen_subset=test \
#         override.data=${DATA_PATH} \
#         override.label_dir=${DATA_PATH} \
#         override.llm_ckpt_path=${LLM_PATH} \
#         generation.beam=20 \
#         generation.no_repeat_ngram_size=3 \
#         dataset.max_tokens=3000 \
#         common_eval.path=${MODEL_PATH} \
#         common_eval.quiet=false \
#         common_eval.results_path=${OUT_PATH}
