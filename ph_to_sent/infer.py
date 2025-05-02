import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import os
import re
from g2p_en import G2p

# Initialize g2p converter
g2p = G2p()

def sentence_to_phonemes(sentence):
    # g2p returns a list with phonemes and spaces for word boundaries
    phonemes = g2p(sentence)
    # Remove spaces (word boundaries) and join phonemes with space
    phonemes = [ph for ph in phonemes if ph != ' ']
    return ' '.join(phonemes)

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
            base_model_path = "/home/rijain@ad.mee.tcd.ie/Experiments/proj/VSR-LLM/checkpoints/Llama-2-7b-hf"
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

def phonemes_to_sentence(model, tokenizer, phoneme_seq):
    if isinstance(phoneme_seq, list):
        phoneme_str = " ".join(phoneme_seq)
    else:
        phoneme_str = phoneme_seq
    prompt = f"Translate the following phonemes into text: {phoneme_str}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated sentence (removing the prompt)
    result = generated_text[len(prompt):].strip()
    
    # # Clean any remaining formatting artifacts
    # # Remove section markers and other wiki formatting
    # result = re.sub(r'=+', '', result)
    # result = re.sub(r'\{\{.*?\}\}', '', result)
    # result = re.sub(r'\[\[.*?\]\]', '', result)
    # # Remove HTML/XML tags
    # result = re.sub(r'<.*?>', '', result)
    # # Remove URLs
    # result = re.sub(r'https?://\S+', '', result)
    # # Remove references
    # result = re.sub(r'<ref.*?>.*?</ref>', '', result)
    # # Remove list markers
    # result = re.sub(r'^\s*[\*#]+\s*', '', result)
    # # Remove bold/italics markup
    # result = re.sub(r"'''?.*?'''?", lambda m: m.group(0).replace("'", ""), result)
    # # Clean multiple spaces and newlines
    # result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert phoneme sequences to sentences")
    parser.add_argument("--model_dir", type=str, default="./llama2_phoneme2sentence_g2p/checkpoint-1000",
                        help="Directory containing the fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, 
                        default="/home/rijain@ad.mee.tcd.ie/Experiments/proj/VSR-LLM/checkpoints/Llama-2-7b-hf",
                        help="Path to the base model (needed for LoRA models)")
    parser.add_argument("--phonemes", type=str,
                        default="DH AH K AE M ER AH IH S",
                        help="Phoneme sequence to convert (space-separated)")
    parser.add_argument("--sentence", type=str,
                        default=None,
                        help="English sentence to convert to phonemes and back (for validation)")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_dir, args.base_model)
    
    # Process an English sentence if provided
    if args.sentence is not None:
        original_sentence = args.sentence
        print("\nValidating full pipeline:")
        print(f"Original sentence: {original_sentence}")
        
        # Convert sentence to phonemes
        phonemes = sentence_to_phonemes(original_sentence)
        print(f"Generated phonemes: {phonemes}")
        
        # Convert phonemes back to text
        reconstructed = phonemes_to_sentence(model, tokenizer, phonemes)
        print(f"Reconstructed sentence: {reconstructed}")
        
        # Calculate similarity (just a simple word overlap percentage)
        original_words = set(original_sentence.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        if original_words:
            overlap = len(original_words.intersection(reconstructed_words)) / len(original_words)
            print(f"Word overlap: {overlap:.2%}")
    
    # Process phoneme sequence if provided
    elif args.phonemes:
        phoneme_seq = args.phonemes.split()
        result = phonemes_to_sentence(model, tokenizer, phoneme_seq)
        print("\nResults:")
        print(f"Phonemes: {args.phonemes}")
        print(f"Predicted sentence: {result}")
    
    else:
        print("Please provide either --phonemes or --sentence")

if __name__ == "__main__":
    main()


## usage: python infer.py --phonemes "DH AH K AE M ER AH IH S"
## usage: python infer.py --sentence "Hello, how are you?"
## python infer.py --model_dir ./train_phon/final/ --base_model /home/rijain@ad.mee.tcd.ie/Experiments/proj/VSR-LLM/checkpoints/Llama-2-7b-hf --sentence "Human beings are completed creatures"