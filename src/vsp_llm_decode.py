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

# Import BERTScore for semantic evaluation
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore not available. To enable, install: pip install bert-score")

# Import Word Mover's Distance - simpler version without external dependencies
try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    WMD_AVAILABLE = True
    # Pre-download the model or use a cached version
    wmd_model = None
except ImportError:
    WMD_AVAILABLE = False
    print("Word Mover's Distance not available. To enable, install: pip install gensim")

# Import METEOR Score
try:
    from nltk.translate.meteor_score import single_meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    print("METEOR score not available. To enable, install: pip install nltk")

# Import for Semantic Embedding Distance
try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    SEMANTIC_WER_AVAILABLE = True
    semantic_model = None
except ImportError:
    SEMANTIC_WER_AVAILABLE = False
    print("Semantic WER not available. To enable, install: pip install sentence-transformers")

# Import ROUGE Score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("ROUGE score not available. To enable, install: pip install rouge-score")

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

# Function to calculate BERTScore
def calculate_bertscore(hypos, refs, model_type="microsoft/deberta-xlarge-mnli"):
    """
    Calculate BERTScore for hypothesis and reference texts
    Using state-of-the-art DeBERTa model for semantic similarity
    """
    if not BERTSCORE_AVAILABLE:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    try:
        # Convert inputs to ensure they're strings
        hypos = [str(h) for h in hypos]
        refs = [str(r) for r in refs]
        
        # Use a simpler model for robustness
        model_type = "roberta-base"  # More stable than DeBERTa for this use case
        
        # Use state-of-the-art model
        P, R, F1 = bert_score(hypos, refs, lang="en", model_type=model_type, 
                             rescale_with_baseline=True)
        
        # Return mean scores
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item())
        }
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {str(e)}")
        return {"precision": 0, "recall": 0, "f1": 0}

# Function to calculate Word Mover's Distance (WMD) - simplified version
def calculate_wmd(hypo_str, ref_str):
    """
    Calculate word similarity between hypothesis and reference texts.
    Uses Jaccard similarity (overlap/union) of word sets.
    
    Returns:
        float: Similarity score (0-1, higher is better) or -1 if unavailable
    """
    global wmd_model
    
    if not WMD_AVAILABLE:
        return -1
    
    try:
        # Simple word overlap calculation as fallback
        hypo_tokens = set(w.lower() for w in hypo_str.strip().split())
        ref_tokens = set(w.lower() for w in ref_str.strip().split())
        
        # Jaccard similarity
        if not hypo_tokens or not ref_tokens:
            return 0
            
        overlap = len(hypo_tokens.intersection(ref_tokens))
        union = len(hypo_tokens.union(ref_tokens))
        
        # Similarity score (0 to 1, higher is better)
        similarity = overlap / union if union > 0 else 0
        return similarity
        
    except Exception as e:
        logger.error(f"Error calculating word similarity: {str(e)}")
        return -1

# Calculate METEOR score between hypothesis and reference
def calculate_meteor(hypo_str, ref_str):
    """
    Calculate METEOR score between hypothesis and reference.
    METEOR considers synonyms, stemming, and paraphrases.
    
    Returns:
        float: METEOR score (0-1, higher is better)
    """
    if not METEOR_AVAILABLE:
        return 0
    
    try:
        # Check for empty strings
        if not hypo_str.strip() or not ref_str.strip():
            return 0
            
        # Calculate METEOR score
        return single_meteor_score(ref_str.split(), hypo_str.split())
    except Exception as e:
        logger.error(f"Error calculating METEOR score: {str(e)}")
        return 0

# Calculate Word Error Rate with Semantic Embedding Distance
def calculate_semantic_wer(hypo_str, ref_str):
    """
    Calculate Word Error Rate with Semantic Embedding Distance.
    Uses sentence embeddings to calculate semantic similarity between words.
    
    Returns:
        dict: Semantic WER score and similarity
    """
    global semantic_model
    
    if not SEMANTIC_WER_AVAILABLE:
        return {"sem_wer": 100, "similarity": 0}
    
    try:
        # Load model once for efficiency
        if semantic_model is None:
            logger.info("Loading sentence transformer model for semantic WER...")
            semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Better model for similarity
            logger.info("Sentence transformer model loaded.")
        
        # Check for empty strings
        if not hypo_str.strip() or not ref_str.strip():
            return {"sem_wer": 100, "similarity": 0}
        
        # Get embeddings for whole sentences
        hypo_emb = semantic_model.encode(hypo_str, convert_to_tensor=True)
        ref_emb = semantic_model.encode(ref_str, convert_to_tensor=True)
        
        # Calculate cosine similarity between sentences
        similarity = F.cosine_similarity(hypo_emb.unsqueeze(0), ref_emb.unsqueeze(0))[0].item()
        
        # Semantic WER: Lower is better, so we use (1 - similarity) * 100
        # This scales from 0 (identical) to 100 (completely different)
        sem_wer = (1 - similarity) * 100
        
        return {"sem_wer": sem_wer, "similarity": similarity}
    except Exception as e:
        logger.error(f"Error calculating semantic WER: {str(e)}")
        return {"sem_wer": 100, "similarity": 0}

# Function to calculate ROUGE scores
def calculate_rouge(hypo_str, ref_str):
    """
    Calculate ROUGE scores between hypothesis and reference.
    ROUGE measures overlap of n-grams, word sequences, and word pairs.
    
    Returns:
        dict: ROUGE scores (precision, recall, f1) for ROUGE-1, ROUGE-2, and ROUGE-L
    """
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    try:
        # Check for empty strings
        if not hypo_str.strip() or not ref_str.strip():
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        # Initialize ROUGE scorer with desired metrics
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate ROUGE scores
        scores = scorer.score(ref_str, hypo_str)
        
        # Extract F1 scores for simplicity
        result = {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
        
        return result
    except Exception as e:
        logger.error(f"Error calculating ROUGE score: {str(e)}")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

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
    
    # Set padding side to 'left' for Qwen models as required
    is_qwen_model = 'qwen' in cfg.override.llm_ckpt_path.lower()
    if is_qwen_model:
        logger.info("Setting tokenizer padding_side to 'left' for Qwen model")
        tokenizer.padding_side = 'left'

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
            model.half()

    # Load dataset (possibly sharded)
    cfg.dataset.batch_size = 4
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
    result_dict = {'utt_id': [], 'ref': [], 'hypo': [], 'instruction': [], 'wer': [], 'cer': []}
    if hasattr(model, 'cfg') and model.cfg.use_ctc:
        result_dict['ctc_wer'] = []
    model = models[0]
    for sample_idx, sample in enumerate(progress):
        # Simple log message for each group of samples
        
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        sample['net_input']['source']['video'] = sample['net_input']['source']['video'].to(torch.half)
        
        # Get CTC emissions if enabled
        ctc_decoded = []
        if hasattr(model, 'cfg') and model.cfg.use_ctc:
            model.cfg.use_ctc = False   #FORCE CTC OFF
            output = model.encoder(**sample["net_input"])
            
            # Ensure CTC head projector is on the same device as features
            if use_cuda:
                if hasattr(model, 'ctc_head_projector'):
                    model.ctc_head_projector = model.ctc_head_projector.cuda()
                if hasattr(model, 'ctc_head_encoder'):
                    model.ctc_head_encoder = model.ctc_head_encoder.cuda()
            
            try:
                ctc_log_probs = model.get_ctc_emissions(output, model.cfg.ctc_feature_source)
                
                # Make sure tensor is in [T, B, V] format for CTC decoding
                if ctc_log_probs.dim() == 3 and ctc_log_probs.size(0) != output['encoder_out'].size(1):
                    # Transpose if necessary [B, T, V] -> [T, B, V]
                    ctc_log_probs = ctc_log_probs.transpose(0, 1)
                
                ctc_decoded = model.decode_ctc(ctc_log_probs)
                
                # Simply report number of CTC outputs
                logger.info(f"CTC decoding: {len(ctc_decoded)} outputs")
                
                # Ensure CTC outputs are not empty
                for i, text in enumerate(ctc_decoded):
                    if len(text.strip()) < 2:  # If text is too short (empty or just whitespace)
                        ctc_decoded[i] = "no valid ctc output"
            except Exception as e:
                logger.error(f"Error during CTC decoding: {str(e)}")
                ctc_decoded = []
        
        eos_token_id = tokenizer.convert_tokens_to_ids("Transcript:")
        
        
        # Temporarily disable CTC for generation to avoid double processing
        use_ctc_original = None
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'use_ctc'):
            use_ctc_original = model.cfg.use_ctc
            model.cfg.use_ctc = False
            
        best_hypo = model.generate(
            target_list=sample["target"], 
            num_beams=cfg.generation.beam,
            max_length=cfg.generation.max_len_b,
            min_length=cfg.generation.min_len,
            length_penalty=cfg.generation.lenpen,
            no_repeat_ngram_size=cfg.generation.no_repeat_ngram_size,
            temperature=0.1,
            repetition_penalty=1.5,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            **sample["net_input"]
        )
        
        # Restore original CTC setting
        if use_ctc_original is not None:
            model.cfg.use_ctc = use_ctc_original
            
        logger.info("--------------------------LLM generation complete----------------------------")
        
        import re

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
            
        best_hypo = tokenizer.batch_decode(
                best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        #best_hypo = [post_process(hyp) for hyp in best_hypo]
        #best_hypo = [clean_hyp_output(hyp) for hyp in best_hypo]
        #best_hypo = [hyp.rstrip('#').rstrip('!') for hyp in best_hypo]
        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            target = sample['target'][i].masked_fill(
                sample['target'][i] == -100, 0
            )
            ref_sent = tokenizer.decode(target.int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #ref_sent = ref_sent.rstrip('#').rstrip('!')
            #ref_sent = post_process(ref_sent)
            #ref_sent = clean_hyp_output(ref_sent)
            result_dict['ref'].append(ref_sent)
            hypo_str = best_hypo[i]
            instruction = tokenizer.decode(sample['net_input']['source']['text'][i].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['instruction'].append(instruction)
            result_dict['hypo'].append(hypo_str)
            
            # Calculate per-sample metrics
            hypo_words, ref_words = hypo_str.strip().split(), ref_sent.strip().split()
            sample_wer = 100 * editdistance.eval(hypo_words, ref_words) / len(ref_words) if len(ref_words) > 0 else 0
            sample_cer = 100 * calculate_cer(hypo_str, ref_sent)

            # Calculate per-sample BLEU score
            try:
                sample_bleu = sacrebleu.sentence_bleu(hypo_str, [ref_sent]).score
                if 'bleu' not in result_dict:
                    result_dict['bleu'] = []
                result_dict['bleu'].append(sample_bleu)
            except Exception as e:
                logger.error(f"Error computing sample BLEU score: {str(e)}")
                sample_bleu = 0
            
            # Calculate per-sample BERTScore
            sample_bertscore_f1 = 0
            if BERTSCORE_AVAILABLE:
                try:
                    # Only calculate BERTScore for non-empty outputs
                    if len(hypo_str.strip()) > 0 and len(ref_sent.strip()) > 0:
                        sample_bertscore = calculate_bertscore([hypo_str], [ref_sent])
                        sample_bertscore_f1 = sample_bertscore['f1']
                        
                        # Store all BERTScore components
                        for component in ['precision', 'recall', 'f1']:
                            if f'bertscore_{component}' not in result_dict:
                                result_dict[f'bertscore_{component}'] = []
                            result_dict[f'bertscore_{component}'].append(sample_bertscore[component])
                except Exception as e:
                    logger.error(f"Error computing sample BERTScore: {str(e)}")
            
            # Calculate per-sample WordSim
            sample_word_sim = 0
            if WMD_AVAILABLE:
                try:
                    # Only calculate for non-empty outputs
                    if len(hypo_str.strip()) > 0 and len(ref_sent.strip()) > 0:
                        sample_word_sim = calculate_wmd(hypo_str, ref_sent)
                        if 'word_sim' not in result_dict:
                            result_dict['word_sim'] = []
                        if sample_word_sim >= 0:
                            result_dict['word_sim'].append(sample_word_sim)
                except Exception as e:
                    logger.error(f"Error computing word similarity: {str(e)}")
            
            # Calculate per-sample METEOR score
            sample_meteor = 0
            if METEOR_AVAILABLE:
                try:
                    sample_meteor = calculate_meteor(hypo_str, ref_sent)
                    if 'meteor' not in result_dict:
                        result_dict['meteor'] = []
                    result_dict['meteor'].append(sample_meteor)
                except Exception as e:
                    logger.error(f"Error computing METEOR score: {str(e)}")
            
            # Calculate per-sample ROUGE scores
            sample_rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            if ROUGE_AVAILABLE:
                try:
                    # Only calculate for non-empty outputs
                    if len(hypo_str.strip()) > 0 and len(ref_sent.strip()) > 0:
                        sample_rouge_scores = calculate_rouge(hypo_str, ref_sent)
                        
                        # Store ROUGE scores
                        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                            if rouge_type not in result_dict:
                                result_dict[rouge_type] = []
                            result_dict[rouge_type].append(sample_rouge_scores[rouge_type])
                except Exception as e:
                    logger.error(f"Error computing ROUGE scores: {str(e)}")
            
            # Calculate per-sample Semantic WER
            sample_sem_wer = 100
            sample_sem_sim = 0
            if SEMANTIC_WER_AVAILABLE:
                try:
                    sem_result = calculate_semantic_wer(hypo_str, ref_sent)
                    sample_sem_wer = sem_result['sem_wer']
                    sample_sem_sim = sem_result['similarity']
                    
                    if 'sem_wer' not in result_dict:
                        result_dict['sem_wer'] = []
                    if 'sem_sim' not in result_dict:
                        result_dict['sem_sim'] = []
                        
                    result_dict['sem_wer'].append(sample_sem_wer)
                    result_dict['sem_sim'].append(sample_sem_sim)
                except Exception as e:
                    logger.error(f"Error computing Semantic WER: {str(e)}")
            
            # Calculate CTC WER if available
            if hasattr(model, 'cfg') and model.cfg.use_ctc and i < len(ctc_decoded):
                ctc_hypo = ctc_decoded[i]
                
                # Handle empty or whitespace-only CTC outputs
                if ctc_hypo.strip() == "":
                    ctc_hypo = "no valid ctc output"
                    
                ctc_hypo_words = ctc_hypo.strip().split()
                
                # Only calculate WER if we have words to compare
                if len(ctc_hypo_words) > 0 and len(ref_words) > 0:
                    ctc_wer = 100 * editdistance.eval(ctc_hypo_words, ref_words) / len(ref_words)
                else:
                    ctc_wer = 100.0  # If no words to compare, set to 100% error
                
                # Ensure ctc_wer key exists
                if 'ctc_wer' not in result_dict:
                    result_dict['ctc_wer'] = []
                    
                result_dict['ctc_wer'].append(ctc_wer)
                logger.info(f"CTC WER: {ctc_wer:.2f}%, CTC text: '{ctc_hypo}'")
            
            # Store metrics in result dictionary
            result_dict['wer'].append(sample_wer)
            result_dict['cer'].append(sample_cer)
            
            # Log with metrics included
            log_msg = f"\n  REF: {ref_sent}"
            log_msg += f"\n  LLM: {hypo_str}"
            log_msg += f"\n  LLM WER: {sample_wer:.1f}%  CER: {sample_cer:.1f}%"
            
            # Add semantic metrics to log if available
            semantic_metrics = []
            if BERTSCORE_AVAILABLE and sample_bertscore_f1 > 0:
                # Include precision and recall along with F1
                if 'bertscore_precision' in result_dict and 'bertscore_recall' in result_dict:
                    sample_precision = result_dict['bertscore_precision'][-1]
                    sample_recall = result_dict['bertscore_recall'][-1]
                    semantic_metrics.append(f"BERTScore: P={sample_precision:.4f}, R={sample_recall:.4f}, F1={sample_bertscore_f1:.4f}")
                else:
                    semantic_metrics.append(f"BERTScore: {sample_bertscore_f1:.4f}")
            if WMD_AVAILABLE and sample_word_sim > 0:
                semantic_metrics.append(f"WordSim: {sample_word_sim:.4f} (word overlap)")
            if METEOR_AVAILABLE and sample_meteor > 0:
                semantic_metrics.append(f"METEOR: {sample_meteor:.4f}")
            if ROUGE_AVAILABLE and sample_rouge_scores["rougeL"] > 0:
                semantic_metrics.append(f"ROUGE-L: {sample_rouge_scores['rougeL']:.4f}")
            if SEMANTIC_WER_AVAILABLE and sample_sem_sim > 0:
                semantic_metrics.append(f"SemWER: {sample_sem_wer:.1f}% (sim={sample_sem_sim:.4f})")
            if sample_bleu > 0:
                semantic_metrics.append(f"BLEU: {sample_bleu:.2f}")
            
            if semantic_metrics:
                log_msg += f"  ({', '.join(semantic_metrics)})"
            
            if hasattr(model, 'cfg') and model.cfg.use_ctc and i < len(ctc_decoded):
                log_msg += f"\n  CTC: {ctc_hypo}"
                log_msg += f"\n  CTC WER: {ctc_wer:.1f}%"
                
            logger.info(log_msg)

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
        
        # Calculate overall CTC WER if available
        ctc_wer = 0
        if 'ctc_wer' in result_dict and len(result_dict['ctc_wer']) > 0:
            ctc_wer = sum(result_dict['ctc_wer']) / len(result_dict['ctc_wer'])
            logger.info(f"Overall CTC WER: {ctc_wer:.2f}%")
        else:
            logger.info("No CTC WER data available")
        
        # Calculate overall BERTScore if available
        bertscore_avg = 0
        if 'bertscore_f1' in result_dict and len(result_dict['bertscore_f1']) > 0:
            bertscore_avg = sum(result_dict['bertscore_f1']) / len(result_dict['bertscore_f1'])
            logger.info(f"Average BERTScore F1: {bertscore_avg:.4f}")
        
        # Calculate overall word similarity if available
        word_sim_avg = -1
        if 'word_sim' in result_dict and len(result_dict['word_sim']) > 0:
            word_sim_avg = sum(result_dict['word_sim']) / len(result_dict['word_sim'])
            logger.info(f"Average word similarity: {word_sim_avg:.4f}")
        
        # Calculate overall METEOR score if available
        meteor_avg = 0
        if 'meteor' in result_dict and len(result_dict['meteor']) > 0:
            meteor_avg = sum(result_dict['meteor']) / len(result_dict['meteor'])
            logger.info(f"Average METEOR score: {meteor_avg:.4f}")
        
        # Calculate overall ROUGE scores if available
        rouge_avgs = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        if ROUGE_AVAILABLE:
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                if rouge_type in result_dict and len(result_dict[rouge_type]) > 0:
                    rouge_avgs[rouge_type] = sum(result_dict[rouge_type]) / len(result_dict[rouge_type])
                    logger.info(f"Average {rouge_type.upper()} score: {rouge_avgs[rouge_type]:.4f}")
        
        # Calculate overall Semantic WER if available
        sem_wer_avg = 0
        sem_sim_avg = 0
        if 'sem_wer' in result_dict and len(result_dict['sem_wer']) > 0:
            sem_wer_avg = sum(result_dict['sem_wer']) / len(result_dict['sem_wer'])
            sem_sim_avg = sum(result_dict['sem_sim']) / len(result_dict['sem_sim'])
            logger.info(f"Average Semantic WER: {sem_wer_avg:.2f}% (similarity={sem_sim_avg:.4f})")
        
        # Calculate overall BLEU score
        bleu_avg = 0
        if 'bleu' in result_dict and len(result_dict['bleu']) > 0:
            bleu_avg = sum(result_dict['bleu']) / len(result_dict['bleu'])
            logger.info(f"Average BLEU score: {bleu_avg:.2f}")
        
        # Always calculate corpus BLEU score directly
        corpus_bleu = sacrebleu.corpus_bleu(result_dict['hypo'], [result_dict['ref']])
        corpus_bleu_score = corpus_bleu.score
        logger.info(f"Corpus BLEU score: {corpus_bleu_score:.2f}")
        
        # Write results to file
        wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
        with open(wer_fn, "w") as fo:
            fo.write(f"WER: {wer:.2f}%\n")
            fo.write(f"CER: {cer:.2f}%\n")
            if 'ctc_wer' in result_dict and len(result_dict['ctc_wer']) > 0:
                fo.write(f"CTC WER: {ctc_wer:.2f}%\n")
            
            # Add average BERTScore from per-sample calculations
            if bertscore_avg > 0:
                fo.write(f"Average BERTScore F1: {bertscore_avg:.4f}\n")
            
            # Add average word similarity from per-sample calculations
            if word_sim_avg >= 0:
                fo.write(f"Average word similarity: {word_sim_avg:.4f}\n")
            
            # Add average METEOR score from per-sample calculations
            if meteor_avg > 0:
                fo.write(f"Average METEOR score: {meteor_avg:.4f}\n")
            
            # Add average ROUGE scores from per-sample calculations
            if ROUGE_AVAILABLE:
                if rouge_avgs['rouge1'] > 0:
                    fo.write(f"Average ROUGE-1 score: {rouge_avgs['rouge1']:.4f}\n")
                if rouge_avgs['rouge2'] > 0:
                    fo.write(f"Average ROUGE-2 score: {rouge_avgs['rouge2']:.4f}\n")
                if rouge_avgs['rougeL'] > 0:
                    fo.write(f"Average ROUGE-L score: {rouge_avgs['rougeL']:.4f}\n")
            
            # Add average Semantic WER from per-sample calculations
            if sem_wer_avg > 0:
                fo.write(f"Average Semantic WER: {sem_wer_avg:.2f}%\n")
                fo.write(f"Average Semantic Similarity: {sem_sim_avg:.4f}\n")
            
            # Add BLEU scores
            if bleu_avg > 0:
                fo.write(f"Average sentence BLEU score: {bleu_avg:.2f}\n")
            fo.write(f"Corpus BLEU score: {corpus_bleu_score:.2f}\n")
            
            fo.write(f"WER err / num_ref_words = {n_err} / {n_total}\n")
            fo.write(f"CER err / num_ref_chars = {n_char_err} / {n_char_total}\n\n")
            
            # Calculate and report BERTScore if available
            if BERTSCORE_AVAILABLE:
                try:
                    # Calculate corpus-level BERTScore only if we have the per-sample scores
                    if 'bertscore_f1' in result_dict and len(result_dict['bertscore_f1']) > 0:
                        # Use the average of per-sample scores which is more reliable
                        bertscore_precision = sum(result_dict.get('bertscore_precision', [0])) / len(result_dict['bertscore_f1']) if 'bertscore_precision' in result_dict else 0
                        bertscore_recall = sum(result_dict.get('bertscore_recall', [0])) / len(result_dict['bertscore_f1']) if 'bertscore_recall' in result_dict else 0
                        bertscore_f1 = sum(result_dict['bertscore_f1']) / len(result_dict['bertscore_f1'])
                        
                        fo.write(f"BERTScore Precision: {bertscore_precision:.4f}\n")
                        fo.write(f"BERTScore Recall: {bertscore_recall:.4f}\n")
                        fo.write(f"BERTScore F1: {bertscore_f1:.4f}\n\n")
                        logger.info(f"BERTScore: P={bertscore_precision:.4f}, R={bertscore_recall:.4f}, F1={bertscore_f1:.4f} (semantic similarity)")
                except Exception as e:
                    logger.error(f"Error computing corpus BERTScore: {str(e)}")
            
            fo.write(f"{yaml_str}")
            
        # Log overall metrics with more details
        logger.info(f"WER: {wer:.2f}%")
        logger.info(f"CER: {cer:.2f}%")
        
        # More descriptive BERTScore logging
        if 'bertscore_f1' in result_dict and len(result_dict['bertscore_f1']) > 0:
            bertscore_precision = sum(result_dict.get('bertscore_precision', [0])) / len(result_dict['bertscore_f1']) if 'bertscore_precision' in result_dict else 0
            bertscore_recall = sum(result_dict.get('bertscore_recall', [0])) / len(result_dict['bertscore_f1']) if 'bertscore_recall' in result_dict else 0
            bertscore_f1 = sum(result_dict['bertscore_f1']) / len(result_dict['bertscore_f1'])
            logger.info(f"BERTScore: P={bertscore_precision:.4f}, R={bertscore_recall:.4f}, F1={bertscore_f1:.4f} (semantic similarity)")
        
        # More descriptive WordSim logging
        if 'word_sim' in result_dict and len(result_dict['word_sim']) > 0:
            word_sim_avg = sum(result_dict['word_sim']) / len(result_dict['word_sim'])
            logger.info(f"WordSim: {word_sim_avg:.4f} (proportion of shared words)")
        
        # METEOR score logging
        if 'meteor' in result_dict and len(result_dict['meteor']) > 0:
            meteor_avg = sum(result_dict['meteor']) / len(result_dict['meteor'])
            logger.info(f"METEOR: {meteor_avg:.4f} (considers synonyms and paraphrases)")
        
        # ROUGE score logging
        if ROUGE_AVAILABLE:
            if 'rougeL' in result_dict and len(result_dict['rougeL']) > 0:
                rougeL_avg = sum(result_dict['rougeL']) / len(result_dict['rougeL'])
                logger.info(f"ROUGE-L: {rougeL_avg:.4f} (longest common subsequence)")
        
        # Semantic WER logging
        if 'sem_wer' in result_dict and len(result_dict['sem_wer']) > 0:
            sem_wer_avg = sum(result_dict['sem_wer']) / len(result_dict['sem_wer'])
            sem_sim_avg = sum(result_dict['sem_sim']) / len(result_dict['sem_sim'])
            logger.info(f"Semantic WER: {sem_wer_avg:.2f}% (similarity={sem_sim_avg:.4f})")
        
        logger.info("Tokenizer name: %s", getattr(tokenizer, "name_or_path", "Unknown"))
    else:
        # BLEU score is already calculated in the main metrics section above
        # Just write to the dedicated BLEU file for compatibility
        bleu_fn = f"{cfg.common_eval.results_path}/bleu.{fid}"
        with open(bleu_fn, "w") as fo:
            fo.write(f"BLEU: {corpus_bleu_score:.2f}\n")
            if bleu_avg > 0:
                fo.write(f"Average sentence BLEU: {bleu_avg:.2f}\n")
            fo.write(f"{yaml_str}")
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