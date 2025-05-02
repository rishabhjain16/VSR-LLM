import os
import random
import torch
import nltk
import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import re
import argparse

from datasets import Dataset
from g2p_en import G2p

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a phoneme-to-text model")
parser.add_argument("--output_dir", type=str, default="./llama2_phoneme2sentence_model",
                    help="Directory to save model checkpoints and final model")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size per device")
args = parser.parse_args()

# Define output directory (used consistently throughout)
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download CMUdict if not already done
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import cmudict

arpabet = cmudict.dict()

g2p = G2p()

def sentence_to_phonemes(sentence):
    try:
        # g2p returns a list with phonemes and spaces for word boundaries
        phonemes = g2p(sentence)
        # Remove spaces (word boundaries) and join phonemes with space
        phonemes = [ph for ph in phonemes if ph != ' ']
        return ' '.join(phonemes)
    except Exception as e:
        # If any error occurs (number too large, etc.), return None
        # This will allow the calling code to skip this sentence
        print(f"Warning: Could not convert sentence to phonemes: {e}")
        return None

# If want to use IPA phonemes instead of CMUdict which is Arpabet
# from epitran import Epitran
# epi = Epitran('eng-Latn')

# def sentence_to_phonemes_ipa(sentence):
#     # Convert to IPA
#     ipa = epi.transliterate(sentence)
#     # Space between phonemes
#     phonemes = ' '.join(list(ipa))
#     return phonemes
########################################################

# def sentence_to_phonemes(sentence):
#     words = sentence.lower().split()
#     phonemes = []
#     for word in words:
#         # Remove punctuation for lookup
#         word_clean = ''.join([c for c in word if c.isalpha()])
#         if word_clean in arpabet:
#             # Use the first pronunciation
#             phonemes.extend(arpabet[word_clean][0])
#         else:
#             # Skip OOV words
#             continue
#     return ' '.join(phonemes)

# 1. Load WikiText dataset
print("Loading WikiText dataset...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# Clean data to remove WikiText formatting
print("Cleaning WikiText data...")
def clean_wikitext(text):
    # Only remove section headers that are on their own lines
    text = re.sub(r'^=+\s.+\s=+$', '', text, flags=re.MULTILINE)
    
    # Keep the text within links, just remove brackets
    text = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', text)  # Simple links [[link]]
    text = re.sub(r'\[\[[^|]+\|([^\]]+)\]\]', r'\1', text)  # Links with display text [[link|text]]
    
    # Remove templates but keep content where possible
    text = re.sub(r'\{\{[^{}]+\}\}', '', text)  # Simple templates
    
    # Remove HTML tags but keep their content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs but keep domain for context
    text = re.sub(r'https?://([^/\s]+)[^\s]*', r'\1', text)
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    
    # Remove list markers but keep the text
    text = re.sub(r'^\s*[\*#]+\s*', '', text, flags=re.MULTILINE)
    
    # Replace bold/italics with plain text
    text = re.sub(r"'''([^']+)'''", r'\1', text)  # Bold
    text = re.sub(r"''([^']+)''", r'\1', text)   # Italic
    
    # Clean multiple spaces and newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Keep some paragraph structure
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

texts = [clean_wikitext(x['text']) for x in dataset["train"] if x['text'].strip()]
# Filter out empty texts after cleaning
texts = [text for text in texts if text.strip()]

# 2. Convert sentences to phoneme sequences
print("Converting sentences to phonemes...")
data_pairs = []
for sent in texts:
    # Filter very short or very long sentences
    if len(sent.split()) < 3 or len(sent.split()) > 40:
        continue
    phonemes = sentence_to_phonemes(sent)
    if phonemes and phonemes.strip():
        data_pairs.append({'phonemes': phonemes, 'sentence': sent})
    if len(data_pairs) % 1000 == 0 and len(data_pairs) > 0:
        print(f"Processed {len(data_pairs)} sentence-phoneme pairs")

print(f"Created {len(data_pairs)} valid phoneme-sentence pairs")

# 3. Create train/validation split
random.shuffle(data_pairs)
split_idx = int(len(data_pairs) * 0.95)
train_pairs = data_pairs[:split_idx]
val_pairs = data_pairs[split_idx:]

# 4. Load Llama-2-7b with 4-bit quantization
model_name = "/home/rijain@ad.mee.tcd.ie/Experiments/proj/VSR-LLM/checkpoints/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 5. Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)

# 6. Tokenization function with prompt
def preprocess(example):
    # Create instruction prompt format
    prompt = f"Translate the following phonemes into text: {example['phonemes']}"
    # Target with EOS token
    target = f" {example['sentence']}{tokenizer.eos_token}"
    
    # Combine for proper language modeling (complete sequence)
    full_text = prompt + target
    
    # Tokenize the full sequence
    encoded = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Create labels: -100 for prompt tokens (not to be predicted), actual ids for target tokens
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=512)
    prompt_len = len(prompt_tokens["input_ids"])
    
    # Set labels to -100 for the prompt part (we don't want to predict those)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    # Make sure lengths match (padding might have added tokens)
    if len(labels) < len(input_ids):
        labels = labels + [-100] * (len(input_ids) - len(labels))
    elif len(labels) > len(input_ids):
        labels = labels[:len(input_ids)]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = Dataset.from_list(train_pairs)
val_dataset = Dataset.from_list(val_pairs)

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=args.epochs,
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=100,
    report_to="none"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 9. Train
print("Starting training...")
trainer.train()

# 10. Save the model
print("Saving final model...")
final_model_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Training complete. Final model saved to {final_model_path}")

# Save a README with training details
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(f"# Phoneme-to-Text Model\n\n")
    f.write(f"- Training date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- Epochs: {args.epochs}\n")
    f.write(f"- Batch size: {args.batch_size}\n")
    f.write(f"- Training pairs: {len(train_pairs)}\n")
    f.write(f"- Validation pairs: {len(val_pairs)}\n\n")
    f.write(f"## Usage\n\n")
    f.write(f"For inference, use either:\n")
    f.write(f"- Latest checkpoint: `{OUTPUT_DIR}/checkpoint-XXXX`\n")
    f.write(f"- Final model: `{final_model_path}`\n")


##usage python train.py --output_dir ./llama2_phoneme2sentence_model --epochs 1 --batch_size 4