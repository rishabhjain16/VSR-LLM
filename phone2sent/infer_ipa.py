#IPA Inference
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import os
import re
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# Initialize phonemizer backend for English
backend = EspeakBackend('en-us', preserve_punctuation=True)

def sentence_to_ipa(sentence, granularity="char"):
    try:
        # For word-level, we keep word boundaries but also add spaces between phonemes
        if granularity == "word":
            # First get IPA with word boundaries
            phonemes = backend.phonemize([sentence], separator=Separator(word=' ', phone=''))[0]
            
            # Then add spaces between individual phonemes while preserving word boundaries
            words = phonemes.split()
            spaced_words = []
            for word in words:
                # Add spaces between each IPA character within a word
                spaced_word = ' '.join(list(word))
                spaced_words.append(spaced_word)
            
            # Join the words back with extra space to mark word boundaries
            phonemes = '  '.join(spaced_words)  # Double space as word boundary
        else:  # Character-level
            # First get the IPA with no word boundaries
            phonemes = backend.phonemize([sentence], separator=Separator(word='', phone=''))[0]
            # Then add spaces between each IPA character
            phonemes = ' '.join(list(phonemes))
            
        return phonemes
    except Exception as e:
        print(f"Warning: Could not convert sentence to IPA: {e}")
        return None

def load_model(model_dir, base_model_path=None):
    """
    Load a PEFT/LoRA fine-tuned model
    
    Args:
        model_dir: Path to the fine-tuned LoRA adapter
        base_model_path: Path to the base model (needed for tokenizer)
                        If None, will try to extract from the adapter config
    """
    # Determine if this is a LoRA adapter directory
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    if is_lora:
        # For LoRA models, we need both the adapter and base model
        print(f"Loading LoRA adapter from {model_dir}")
        
        # If base_model_path not provided, try to get it from the adapter config
        if base_model_path is None:
            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model_path = peft_config.base_model_name_or_path
            print(f"Using base model: {base_model_path}")
            
        # If base model path still not available or is a remote path, use the original model path
        if base_model_path is None or base_model_path.startswith("http"):
            base_model_path = "/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/Llama-2-7b-hf"
            print(f"Using default base model: {base_model_path}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load the LoRA adapter on top of the base model
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        # For regular models (not LoRA)
        print(f"Loading regular model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    return model, tokenizer

def ipa_to_sentence(model, tokenizer, ipa_seq):
    if isinstance(ipa_seq, list):
        ipa_str = " ".join(ipa_seq)
    else:
        ipa_str = ipa_seq
    prompt = f"Translate the following IPA phonemes into text: {ipa_str}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            temperature=0.2,
            do_sample=True, 
            top_p=0.92,
            top_k=40,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated sentence (removing the prompt)
    result = generated_text[len(prompt):].strip()
    
    # Clean any formatting artifacts
    # Fix contractions by removing spaces around apostrophes
    result = re.sub(r'\s+\'', '\'', result)  # Fix "don 't" → "don't"
    result = re.sub(r'\'\s+', '\'', result)  # Fix "it 's" → "it's"
    
    # Fix spacing around basic punctuation
    result = re.sub(r'\s+([.,?!])', r'\1', result)  # Fix spaces before punctuation
    
    # Fix multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert IPA sequences to sentences")
    parser.add_argument("--model_dir", type=str, default="./trained/wiki2_ipa/checkpoint-1000",
                        help="Directory containing the fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, 
                        default="/home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/Llama-2-7b-hf",
                        help="Path to the base model (needed for LoRA models)")
    parser.add_argument("--ipa", type=str,
                        default="ð ə k æ m ə r ə ɪ z",
                        help="IPA phoneme sequence to convert (space-separated)")
    parser.add_argument("--sentence", type=str,
                        default=None,
                        help="English sentence to convert to IPA and back (for validation)")
    parser.add_argument("--granularity", type=str, choices=["char", "word"], default="char",
                        help="IPA granularity: 'char' for character-level spaces or 'word' for word-level")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_dir, args.base_model)
    
    # Process an English sentence if provided
    if args.sentence is not None:
        original_sentence = args.sentence
        print("\nValidating full pipeline:")
        print(f"Original sentence: {original_sentence}")
        
        # Convert sentence to IPA phonemes
        phonemes = sentence_to_ipa(original_sentence, args.granularity)
        print(f"Generated IPA phonemes: {phonemes}")
        
        # Convert IPA phonemes back to text
        reconstructed = ipa_to_sentence(model, tokenizer, phonemes)
        print(f"Reconstructed sentence: {reconstructed}")
        
        # Calculate similarity (just a simple word overlap percentage)
        original_words = set(original_sentence.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        if original_words:
            overlap = len(original_words.intersection(reconstructed_words)) / len(original_words)
            print(f"Word overlap: {overlap:.2%}")
    
    # Process IPA sequence if provided
    elif args.ipa:
        if " " not in args.ipa:
            # If input doesn't have spaces already, add them between characters
            ipa_seq = " ".join(list(args.ipa))
        else:
            ipa_seq = args.ipa
        
        result = ipa_to_sentence(model, tokenizer, ipa_seq)
        print("\nResults:")
        print(f"IPA Phonemes: {ipa_seq}")
        print(f"Predicted sentence: {result}")
    
    else:
        print("Please provide either --ipa or --sentence")

if __name__ == "__main__":
    main() 