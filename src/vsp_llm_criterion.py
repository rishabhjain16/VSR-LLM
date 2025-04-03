# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
import editdistance

@register_criterion("decoder_only_language_modeling_loss", dataclass=FairseqDataclass)
class decoder_only_language_modeling_loss(FairseqCriterion):
    def __init__(self, task, sentence_avg=False, label_smoothing=0.0, debug=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.padding_idx = task.dictionary.pad() if hasattr(task, 'dictionary') else -100
        self.debug = debug  # Debug flag for troubleshooting

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        # Print sample structure for debugging if needed
        if self.debug:
            print("\n===== Sample Structure =====")
            for key in sample:
                if key == 'net_input':
                    print("net_input keys:", list(sample['net_input'].keys()))
                else:
                    print(f"{key}: {type(sample[key])}")
            print("===========================\n")
        
        # Use a unified call pattern for all models
        model_outputs = model(target_list=sample["target"], **sample["net_input"])
        
        # Handle the unified return format (loss, logits, llm_labels)
        loss, lprobs, llm_labels = model_outputs
        
        # Calculate sample size consistently
        if isinstance(sample["target"], list):
            sample_size = len(sample["target"])  # List style
        else:
            sample_size = sample["target"].size(0)  # Tensor style
        
        # Use our universal accuracy computation method with the provided labels
        n_correct, total = self.compute_accuracy(lprobs, sample, llm_labels)
        
        # Create logging output
        logging_output = {
            "loss": loss.item() if hasattr(loss, "item") else loss.data,
            "ntokens": sample.get("ntokens", total),
            "nsentences": sample_size,
            "sample_size": sample_size,
            "n_correct": utils.item(n_correct.data),
            "total": utils.item(total.data),
        }
        
        # Add WER evaluation for models with tokenizer in evaluation mode
        if not model.training and hasattr(model, 'tokenizer'):
            try:
                n_err = 0
                n_total = 0
                n_char_err = 0  # Add counter for character errors
                n_char_total = 0  # Add counter for total characters
                
                with torch.no_grad():
                    refs = model.tokenizer.batch_decode(sample['target'],
                                                      skip_special_tokens=True, 
                                                      clean_up_tokenization_spaces=False)
                    best_hypo = model.generate(**sample["net_input"], num_beams=5, temperature=0.3)
                    hypos = model.tokenizer.batch_decode(best_hypo, 
                                                        skip_special_tokens=True, 
                                                        clean_up_tokenization_spaces=False)
                    
                for hypo, ref in zip(hypos, refs):
                    # Calculate WER
                    hypo_words, ref_words = hypo.strip().split(), ref.strip().split()
                    n_err += editdistance.eval(hypo_words, ref_words)
                    n_total += len(ref_words)
                    
                    # Calculate CER
                    hypo_chars = "".join(hypo.strip().split())
                    ref_chars = "".join(ref.strip().split())
                    n_char_err += editdistance.eval(hypo_chars, ref_chars)
                    n_char_total += len(ref_chars)
                
                # Store all metrics in logging output
                logging_output["n_err"] = n_err
                logging_output["n_total"] = n_total
                logging_output["n_char_err"] = n_char_err
                logging_output["n_char_total"] = n_char_total
            except Exception as e:
                print(f"Warning: WER/CER calculation failed with error: {e}")
        
        return loss, sample_size, logging_output

    def compute_accuracy_llama2(self, lprobs, sample):
        """Compute accuracy for Llama 2 style models using attention masks"""
        try:
            target = sample['net_input']['prev_output_tokens']
            b, t = target.size()
            mask = sample['target_attn_mask'] == 1
            
            # Get predictions from logits
            preds = lprobs[:,-t:].argmax(2)
            
            # Debug shapes to help diagnose issues
            pred_shape = preds.shape
            mask_shape = mask.shape
            target_shape = target.shape
            
            if self.debug:
                print(f"Shape details - pred: {pred_shape}, mask: {mask_shape}, target: {target_shape}")
            
            # Handle shape mismatches by resizing tensors
            if pred_shape != mask_shape or target_shape != mask_shape:
                print(f"Shape mismatch - pred: {pred_shape}, mask: {mask_shape}, target: {target_shape}")
                
                # Find the minimum length among all tensors
                min_len = min(pred_shape[1], mask_shape[1], target_shape[1])
                
                # Resize all tensors to the same length
                if mask_shape[1] > min_len:
                    mask = mask[:, :min_len]
                
                if pred_shape[1] > min_len:
                    preds = preds[:, :min_len]
                
                if target_shape[1] > min_len:
                    target = target[:, :min_len]
            
            # Calculate accuracy with aligned tensors
            n_correct = torch.sum((preds == target) & mask)
            total = torch.sum(mask)
            
            return n_correct, total
            
        except Exception as e:
            print(f"Error in compute_accuracy_llama2: {e}")
            # Return a fallback value
            device = lprobs.device if hasattr(lprobs, 'device') else torch.device('cpu')
            return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    
    def compute_accuracy_llama3(self, lprobs, llm_labels):
        """Compute accuracy for Llama 3 style models using shifted labels"""
        try:
            shifted_logits = lprobs[:, :-1, :]
            shifted_labels = llm_labels[:, 1:]      
            
            predictions = torch.argmax(shifted_logits, dim=-1)
            mask = shifted_labels != -100
            
            correct_predictions = (predictions == shifted_labels) & mask
            
            n_correct = correct_predictions.sum().float()
            total = mask.sum().float()
            return n_correct, total
        except Exception as e:
            print(f"Warning: Error in compute_accuracy_llama3: {e}")
            return torch.tensor(0.0, device=lprobs.device), torch.tensor(1.0, device=lprobs.device)

    def compute_accuracy(self, lprobs, sample, llm_labels=None):
        """
        Universal accuracy computation method that adapts to different model architectures.
        This method will detect the model type and use the appropriate accuracy calculation.
        """
        if self.debug:
            # Print shape information for debugging purposes
            model_info = "Using "
            if llm_labels is not None:
                model_info += "Llama3/Qwen-style with shifted labels"
            elif 'target_attn_mask' in sample and 'prev_output_tokens' in sample['net_input']:
                model_info += "Llama2-style with attention mask"
            else:
                model_info += "generic approach (no specific markers detected)"
            print(f"Model detection: {model_info}")
            
            # Print shapes for debugging
            if 'target' in sample:
                print(f"target shape: {sample['target'].shape if hasattr(sample['target'], 'shape') else 'list type'}")
            if 'net_input' in sample and 'prev_output_tokens' in sample['net_input']:
                print(f"prev_output_tokens shape: {sample['net_input']['prev_output_tokens'].shape}")
            if 'target_attn_mask' in sample:
                print(f"mask shape: {sample['target_attn_mask'].shape}")
            print(f"lprobs shape: {lprobs.shape if hasattr(lprobs, 'shape') else type(lprobs)}")
            if llm_labels is not None:
                print(f"llm_labels shape: {llm_labels.shape}")
        
        # Method 1: If we have labels provided (Llama3/Qwen approach)
        if llm_labels is not None:
            return self.compute_accuracy_llama3(lprobs, llm_labels)
        
        # Method 2: If we have attention masks (Llama2 approach)
        elif 'target_attn_mask' in sample and 'prev_output_tokens' in sample['net_input']:
            return self.compute_accuracy_llama2(lprobs, sample)
        
        # Method 3: Generic approach if none of the above apply
        else:
            # This handles other model architectures that don't fit the above patterns
            try:
                # Get target from sample
                if isinstance(sample["target"], list):
                    # Convert list to tensor
                    device = lprobs.device
                    target = torch.tensor(sample["target"], device=device)
                else:
                    target = sample["target"]
                
                # Create mask (non-padding positions)
                if hasattr(self, 'padding_idx'):
                    mask = target != self.padding_idx
                else:
                    # Default mask (all tokens)
                    mask = torch.ones_like(target, dtype=torch.bool)
                
                # Get predictions
                preds = lprobs.argmax(dim=-1)
                
                # Adjust shapes to make compatible
                if preds.shape != target.shape:
                    if self.debug:
                        print(f"Shape mismatch in generic approach - preds: {preds.shape}, target: {target.shape}")
                    min_len = min(preds.size(1), target.size(1))
                    preds = preds[:, :min_len]
                    target = target[:, :min_len]
                    if mask.shape[1] > min_len:
                        mask = mask[:, :min_len]
                
                # Calculate accuracy
                n_correct = torch.sum((preds == target) & mask)
                total = torch.sum(mask)
                return n_correct, total
                
            except Exception as e:
                print(f"Warning: Generic accuracy calculation failed: {e}")
                return torch.tensor(0.0, device=lprobs.device), torch.tensor(1.0, device=lprobs.device)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        # Handle WER metrics if available
        n_err = sum(log.get("n_err", 0) for log in logging_outputs)
        n_total = sum(log.get("n_total", 0) for log in logging_outputs)
        
        # Handle CER metrics if available
        n_char_err = sum(log.get("n_char_err", 0) for log in logging_outputs)
        n_char_total = sum(log.get("n_char_total", 0) for log in logging_outputs)
        
        if n_total > 0:
            metrics.log_scalar("_n_err", n_err)
            metrics.log_scalar("_n_total", n_total)
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_n_err"].sum * 100.0 / meters["_n_total"].sum, 3
                )
                if meters["_n_total"].sum > 0
                else float("nan"),
            )
            
        if n_char_total > 0:
            metrics.log_scalar("_n_char_err", n_char_err)
            metrics.log_scalar("_n_char_total", n_char_total)
            metrics.log_derived(
                "cer",
                lambda meters: safe_round(
                    meters["_n_char_err"].sum * 100.0 / meters["_n_char_total"].sum, 3
                )
                if meters["_n_char_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
