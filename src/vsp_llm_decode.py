# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#============================ 69 ============================

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import pdb

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    GenerationConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf, MISSING
import sacrebleu

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: ["video"], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    eval_bleu: bool = field(default=False, metadata={'help': 'evaluate bleu score'})
    llm_ckpt_path: str = field(default=MISSING, metadata={'help': 'path to llama checkpoint'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    
    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)

    return _main(cfg, sys.stdout)

from fairseq import tasks
from transformers import AutoTokenizer

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.override.llm_ckpt_path
    )

    # # Ensure the pad token exists; if not, add it
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # # Set pad_token and padding_side separately
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    # tokenizer.truncation_side = "left"

    model_override_cfg = {'model':{'llm_ckpt_path':cfg.override.llm_ckpt_path}}

    # model_override_cfg = {
    #     'model': {
    #         'llm_ckpt_path': cfg.override.llm_ckpt_path,
    #         'generation_params': {
    #             'remove_invalid_values': True,
    #             #'early_stopping': True,
    #             #'no_repeat_ngram_size': 4,
    #             #'max_new_tokens': cfg.generation.max_len_b,
    #             #'min_length': 1,
    #             'num_beams': cfg.generation.beam,
    #             'length_penalty': cfg.generation.lenpen
    #         }
    #     }
    # }
    def post_process_output(hyp):
        # Remove padding tokens
        hyp = hyp.replace(tokenizer.pad_token, "")
        # Remove trailing hashes
        hyp = hyp.rstrip('#')
        return hyp
    
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path],model_override_cfg,strict=False)
    task.post_process = post_process_output
    models = [model.eval() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None :
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.llm_ckpt_path = cfg.override.llm_ckpt_path
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    
    # Log information about the LLM model being used
    model_path = cfg.override.llm_ckpt_path
    model_name = os.path.basename(model_path)
    logger.info(f"Using LLM model: {model_name} from path: {model_path}")
    
    task.load_dataset('test', task_cfg=cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.encoder.cuda()
            model.avfeat_to_llm.cuda()
            
            # Also move CTC heads to GPU if they exist
            if hasattr(model, 'ctc_head_encoder') and model.ctc_head_encoder is not None:
                model.ctc_head_encoder.cuda()
            if hasattr(model, 'ctc_head_projector') and model.ctc_head_projector is not None:
                model.ctc_head_projector.cuda()
                
            model.half()

    # Load dataset (possibly sharded)
    cfg.dataset.batch_size = 1
    cfg.dataset.max_tokens = 2000
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()
    def decode_fn(x):
        symbols_ignore = {"<unk>", "<mask>","<pad>", "</s>"}
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return tokenizer.decode(x, skip_special_tokens=True)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words
    
    def post_process(text):
        # Remove trailing '#' and '!'
        return text.rstrip('#').rstrip('!').strip()
    
    # Add function to calculate character error rate
    def calculate_cer(hypo_str, ref_str):
        """Calculate character error rate between hypothesis and reference."""
        # Convert to string if they're not already
        hypo_chars = "".join(hypo_str.strip().split())
        ref_chars = "".join(ref_str.strip().split())
        
        # Calculate edit distance at character level
        edit_distance = editdistance.eval(hypo_chars, ref_chars)
        
        # Avoid division by zero
        if len(ref_chars) == 0:
            return 1.0
        
        return edit_distance / len(ref_chars)
    
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': [], 'instruction': [], 'wer': [], 'cer': [], 'ctc_output': []}
    model = models[0]
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        sample['net_input']['source']['video'] = sample['net_input']['source']['video'].to(torch.half)
        
        # best_hypo = model.generate(
        #     target_list=sample["target"], 
        #     num_beams=cfg.generation.beam, 
        #     length_penalty=cfg.generation.lenpen,
        #     pad_token_id=tokenizer.pad_token_id,       # Correctly set pad_token_id
        #     eos_token_id=tokenizer.eos_token_id,       # Correctly set eos_token_id
        #     attention_mask=sample["net_input"].get("attention_mask"),
        #     **sample["net_input"]
        # )
        eos_token_id = tokenizer.convert_tokens_to_ids("Transcript:")
        best_hypo = model.generate(
            target_list=sample["target"], 
            num_beams=cfg.generation.beam,
            #max_length=cfg.generation.max_len_b,
            #min_length=cfg.generation.min_len,
            length_penalty=cfg.generation.lenpen,
            #no_repeat_ngram_size=cfg.generation.no_repeat_ngram_size,
            #temperature=0.1 ,  # Added temperature
            #repetition_penalty=1.5,  # Increased penalty
            #top_p=0.9,  # Added nucleus sampling
            #pad_token_id=tokenizer.pad_token_id,
            #eos_token_id=eos_token_id,
            #attention_mask=sample["net_input"].get("attention_mask"),
            use_ctc_decoding=True,  # Explicitly enable CTC decoding
            ctc_weight_decode=0.3,  # Set hybrid decoding weight
            return_ctc_outputs=True,  # Return both CTC and LLM outputs
            return_both_outputs=True,  # Return both LLM-only and hybrid outputs
            **sample["net_input"]
        )
        
        # Log explicitly that hybrid approach is activated
        logger.info(f"=== HYBRID RERANKING ENABLED: Weight={0.3} Beam={cfg.generation.beam} ===")
        
        import re

        # Process the outputs based on the return type
        if isinstance(best_hypo, dict):
            llm_output = best_hypo.get('llm_output', None)
            hybrid_output = best_hypo.get('hybrid_output', None)
            ctc_output = best_hypo.get('ctc_output', None)
            
            # Extract the first hypothesis from each if they're batched
            if llm_output is not None and not isinstance(llm_output, list) and llm_output.dim() > 1 and llm_output.size(0) > 1:
                # Take first item from first batch as LLM output
                llm_hypo = llm_output[0].unsqueeze(0) if llm_output.dim() > 1 else llm_output
            else:
                llm_hypo = llm_output
                
            if hybrid_output is not None:
                hybrid_hypo = hybrid_output
            else:
                hybrid_hypo = None
                
            # Ensure these keys exist in result_dict
            if 'llm_only_output' not in result_dict:
                result_dict['llm_only_output'] = []
            if 'llm_only_wer' not in result_dict:
                result_dict['llm_only_wer'] = []
            if 'llm_only_cer' not in result_dict:
                result_dict['llm_only_cer'] = []
            
            if 'hybrid_output' not in result_dict:
                result_dict['hybrid_output'] = []
            if 'hybrid_wer' not in result_dict:
                result_dict['hybrid_wer'] = []
            if 'hybrid_cer' not in result_dict:
                result_dict['hybrid_cer'] = []
            
            # Store CTC outputs in result dictionary
            if ctc_output is not None:
                result_dict['ctc_output'] = ctc_output
        else:
            # Fallback to just using LLM output
            llm_hypo = best_hypo
            hybrid_hypo = None
            
            # Ensure these keys exist in result_dict even in fallback case
            if 'llm_only_output' not in result_dict:
                result_dict['llm_only_output'] = []
            if 'llm_only_wer' not in result_dict:
                result_dict['llm_only_wer'] = []
            if 'llm_only_cer' not in result_dict:
                result_dict['llm_only_cer'] = []
        
        def clean_hyp_output(hyp_text):     
            hyp_text = re.sub(r"[#]+", "", hyp_text)  # Remove '#' artifacts
            sentences = hyp_text.split(". ")  # Split by sentence
            seen = set()
            cleaned_hyp = []

            for sentence in sentences:
                if sentence not in seen:
                    cleaned_hyp.append(sentence)
                    seen.add(sentence)

            return ". ".join(cleaned_hyp).strip()
        
        # Decode both outputs
        llm_hypo_decoded = None
        if llm_hypo is not None:
            llm_hypo_decoded = tokenizer.batch_decode(
                llm_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        hybrid_hypo_decoded = None
        if hybrid_hypo is not None:
            hybrid_hypo_decoded = tokenizer.batch_decode(
                hybrid_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            target = sample['target'][i].masked_fill(
                sample['target'][i] == -100, 0
            )
            ref_sent = tokenizer.decode(target.int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['ref'].append(ref_sent)
            
            instruction = tokenizer.decode(sample['net_input']['source']['text'][i].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['instruction'].append(instruction)
            
            # Process LLM-only output if available
            llm_only_str = ""
            llm_only_wer = None
            llm_only_cer = None
            if llm_hypo_decoded is not None and i < len(llm_hypo_decoded):
                llm_only_str = llm_hypo_decoded[i]
                result_dict['llm_only_output'].append(llm_only_str)
                
                # Calculate WER and CER for LLM-only
                llm_words, ref_words = llm_only_str.strip().split(), ref_sent.strip().split()
                llm_only_wer = 100 * editdistance.eval(llm_words, ref_words) / len(ref_words) if len(ref_words) > 0 else 0
                llm_only_cer = 100 * calculate_cer(llm_only_str, ref_sent)
                result_dict['llm_only_wer'].append(llm_only_wer)
                result_dict['llm_only_cer'].append(llm_only_cer)
            
            # Process hybrid output if available
            hybrid_str = ""
            hybrid_wer = None
            hybrid_cer = None
            if hybrid_hypo_decoded is not None and i < len(hybrid_hypo_decoded):
                hybrid_str = hybrid_hypo_decoded[i]
                result_dict['hybrid_output'].append(hybrid_str)
                result_dict['hypo'].append(hybrid_str)  # For backward compatibility
                
                # Calculate WER and CER for hybrid
                hybrid_words, ref_words = hybrid_str.strip().split(), ref_sent.strip().split()
                hybrid_wer = 100 * editdistance.eval(hybrid_words, ref_words) / len(ref_words) if len(ref_words) > 0 else 0
                hybrid_cer = 100 * calculate_cer(hybrid_str, ref_sent)
                result_dict['hybrid_wer'].append(hybrid_wer)
                result_dict['hybrid_cer'].append(hybrid_cer)
                result_dict['wer'].append(hybrid_wer)  # For backward compatibility
                result_dict['cer'].append(hybrid_cer)  # For backward compatibility
            elif llm_hypo_decoded is not None:
                # Fallback to LLM-only output if hybrid not available
                result_dict['hypo'].append(llm_only_str)
                result_dict['wer'].append(llm_only_wer)
                result_dict['cer'].append(llm_only_cer)
            
            # Include CTC output in log if available
            ctc_output_str = ""
            if 'ctc_output' in result_dict and i < len(result_dict['ctc_output']):
                ctc_output = result_dict['ctc_output'][i]
                ctc_output_str = f"\nCTC:{ctc_output}"
                
                # Calculate WER and CER for CTC output
                ctc_words = ctc_output.strip().split()
                ctc_wer = 100 * editdistance.eval(ctc_words, ref_words) / len(ref_words) if len(ref_words) > 0 else 0
                ctc_cer = 100 * calculate_cer(ctc_output, ref_sent)
                
                # Add CTC metrics to log
                ctc_output_str += f"\nCTC-WER:{ctc_wer:.2f}%\nCTC-CER:{ctc_cer:.2f}%"
                
                # Store CTC metrics in result dictionary if not already there
                if 'ctc_wer' not in result_dict:
                    result_dict['ctc_wer'] = []
                if 'ctc_cer' not in result_dict:
                    result_dict['ctc_cer'] = []
                
                # Ensure lists are the right length
                while len(result_dict['ctc_wer']) < i:
                    result_dict['ctc_wer'].append(None)
                while len(result_dict['ctc_cer']) < i:
                    result_dict['ctc_cer'].append(None)
                
                # Add current metrics
                if i >= len(result_dict['ctc_wer']):
                    result_dict['ctc_wer'].append(ctc_wer)
                else:
                    result_dict['ctc_wer'][i] = ctc_wer
                    
                if i >= len(result_dict['ctc_cer']):
                    result_dict['ctc_cer'].append(ctc_cer)
                else:
                    result_dict['ctc_cer'][i] = ctc_cer
            
            # Log everything together for comparison
            logger.info(f"\nINST:{instruction}\nREF:{ref_sent}")
            
            if llm_only_str:
                logger.info(f"LLM:{llm_only_str}\nLLM-WER:{llm_only_wer:.2f}%\nLLM-CER:{llm_only_cer:.2f}%")
                
            if hybrid_str:
                logger.info(f"HYBRID:{hybrid_str}\nHYBRID-WER:{hybrid_wer:.2f}%\nHYBRID-CER:{hybrid_cer:.2f}%")
                
            logger.info(f"{ctc_output_str}")

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    if not cfg.override.eval_bleu:
        n_err, n_total = 0, 0
        n_char_err, n_char_total = 0, 0
        assert len(result_dict['hypo']) == len(result_dict['ref'])
        for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
            # Calculate WER
            hypo_words, ref_words = hypo.strip().split(), ref.strip().split()
            n_err += editdistance.eval(hypo_words, ref_words)
            n_total += len(ref_words)
            
            # Calculate CER
            hypo_chars = "".join(hypo.strip().split())
            ref_chars = "".join(ref.strip().split())
            n_char_err += editdistance.eval(hypo_chars, ref_chars)
            n_char_total += len(ref_chars)
            
        # Calculate overall WER and CER
        wer = 100 * n_err / n_total if n_total > 0 else 0
        cer = 100 * n_char_err / n_char_total if n_char_total > 0 else 0
        
        # Calculate overall LLM-only WER and CER if available 
        llm_wer, llm_cer = None, None
        llm_n_err, llm_n_total = 0, 0
        llm_n_char_err, llm_n_char_total = 0, 0
        
        if 'llm_only_output' in result_dict and len(result_dict['llm_only_output']) > 0:
            for i, (llm_output, ref) in enumerate(zip(result_dict['llm_only_output'], result_dict['ref'])):
                # Calculate WER
                llm_words, ref_words = llm_output.strip().split(), ref.strip().split()
                llm_n_err += editdistance.eval(llm_words, ref_words)
                llm_n_total += len(ref_words)
                
                # Calculate CER
                llm_chars = "".join(llm_output.strip().split())
                ref_chars = "".join(ref.strip().split())
                llm_n_char_err += editdistance.eval(llm_chars, ref_chars)
                llm_n_char_total += len(ref_chars)
            
            # Calculate overall metrics
            llm_wer = 100 * llm_n_err / llm_n_total if llm_n_total > 0 else 0
            llm_cer = 100 * llm_n_char_err / llm_n_char_total if llm_n_char_total > 0 else 0
        
        # Calculate overall Hybrid WER and CER if available
        hybrid_wer, hybrid_cer = None, None
        hybrid_n_err, hybrid_n_total = 0, 0
        hybrid_n_char_err, hybrid_n_char_total = 0, 0
        
        if 'hybrid_output' in result_dict and len(result_dict['hybrid_output']) > 0:
            for i, (hybrid_output, ref) in enumerate(zip(result_dict['hybrid_output'], result_dict['ref'])):
                # Calculate WER
                hybrid_words, ref_words = hybrid_output.strip().split(), ref.strip().split()
                hybrid_n_err += editdistance.eval(hybrid_words, ref_words)
                hybrid_n_total += len(ref_words)
                
                # Calculate CER
                hybrid_chars = "".join(hybrid_output.strip().split())
                ref_chars = "".join(ref.strip().split())
                hybrid_n_char_err += editdistance.eval(hybrid_chars, ref_chars)
                hybrid_n_char_total += len(ref_chars)
            
            # Calculate overall metrics
            hybrid_wer = 100 * hybrid_n_err / hybrid_n_total if hybrid_n_total > 0 else 0
            hybrid_cer = 100 * hybrid_n_char_err / hybrid_n_char_total if hybrid_n_char_total > 0 else 0
        
        # Calculate overall CTC WER and CER if available
        ctc_wer, ctc_cer = None, None
        ctc_n_err, ctc_n_total = 0, 0
        ctc_n_char_err, ctc_n_char_total = 0, 0
        
        if 'ctc_output' in result_dict and 'ctc_wer' in result_dict:
            for i, (ctc_output, ref) in enumerate(zip(result_dict['ctc_output'], result_dict['ref'])):
                # Calculate WER
                ctc_words, ref_words = ctc_output.strip().split(), ref.strip().split()
                ctc_n_err += editdistance.eval(ctc_words, ref_words)
                ctc_n_total += len(ref_words)
                
                # Calculate CER
                ctc_chars = "".join(ctc_output.strip().split())
                ref_chars = "".join(ref.strip().split())
                ctc_n_char_err += editdistance.eval(ctc_chars, ref_chars)
                ctc_n_char_total += len(ref_chars)
            
            # Calculate overall metrics
            ctc_wer = 100 * ctc_n_err / ctc_n_total if ctc_n_total > 0 else 0
            ctc_cer = 100 * ctc_n_char_err / ctc_n_char_total if ctc_n_char_total > 0 else 0
        
        # Write results to file
        wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
        with open(wer_fn, "w") as fo:
            # If hybrid metrics are available, use them as the primary metrics
            if hybrid_wer is not None:
                fo.write(f"HYBRID-WER: {hybrid_wer:.2f}%\n")
                fo.write(f"HYBRID-CER: {hybrid_cer:.2f}%\n")
                fo.write(f"HYBRID-WER err / num_ref_words = {hybrid_n_err} / {hybrid_n_total}\n")
                fo.write(f"HYBRID-CER err / num_ref_chars = {hybrid_n_char_err} / {hybrid_n_char_total}\n\n")
            
            # Always include the LLM-only metrics for comparison
            if llm_wer is not None:
                fo.write(f"LLM-WER: {llm_wer:.2f}%\n")
                fo.write(f"LLM-CER: {llm_cer:.2f}%\n")
                fo.write(f"LLM-WER err / num_ref_words = {llm_n_err} / {llm_n_total}\n")
                fo.write(f"LLM-CER err / num_ref_chars = {llm_n_char_err} / {llm_n_char_total}\n\n")
                
                # If we have both, show the difference
                if hybrid_wer is not None:
                    fo.write(f"WER DIFF (HYBRID - LLM): {hybrid_wer - llm_wer:.2f}%\n")
                    fo.write(f"CER DIFF (HYBRID - LLM): {hybrid_cer - llm_cer:.2f}%\n\n")
            
            # Include legacy format for backward compatibility
            fo.write(f"WER: {wer:.2f}%\n")
            fo.write(f"CER: {cer:.2f}%\n")
            fo.write(f"WER err / num_ref_words = {n_err} / {n_total}\n")
            fo.write(f"CER err / num_ref_chars = {n_char_err} / {n_char_total}\n\n")
            
            # Add CTC metrics if available
            if ctc_wer is not None and ctc_cer is not None:
                fo.write(f"CTC-WER: {ctc_wer:.2f}%\n")
                fo.write(f"CTC-CER: {ctc_cer:.2f}%\n")
                fo.write(f"CTC-WER err / num_ref_words = {ctc_n_err} / {ctc_n_total}\n")
                fo.write(f"CTC-CER err / num_ref_chars = {ctc_n_char_err} / {ctc_n_char_total}\n\n")
            
            fo.write(f"{yaml_str}")
            
        # Log overall metrics
        if hybrid_wer is not None:
            logger.info(f"HYBRID-WER: {hybrid_wer:.2f}%")
            logger.info(f"HYBRID-CER: {hybrid_cer:.2f}%")
            
        if llm_wer is not None:
            logger.info(f"LLM-WER: {llm_wer:.2f}%")
            logger.info(f"LLM-CER: {llm_cer:.2f}%")
            
        logger.info(f"WER: {wer:.2f}%")
        logger.info(f"CER: {cer:.2f}%")
        
        # Log CTC metrics if available
        if ctc_wer is not None and ctc_cer is not None:
            logger.info(f"CTC-WER: {ctc_wer:.2f}%")
            logger.info(f"CTC-CER: {ctc_cer:.2f}%")
            
        logger.info("Tokenizer name: %s", getattr(tokenizer, "name_or_path", "Unknown"))
    else:
        bleu = sacrebleu.corpus_bleu(result_dict['hypo'], [result_dict['ref']])
        bleu_score = bleu.score
        bleu_fn = f"{cfg.common_eval.results_path}/bleu.{fid}"
        with open(bleu_fn, "w") as fo:
            fo.write(f"BLEU: {bleu_score}\n")
            fo.write(f"{yaml_str}")
        logger.info(f"BLEU: {bleu_score}\n")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()