import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from phonemizer import phonemize
from phonemizer.separator import Separator
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load WikiText dataset
logger.info("Loading WikiText dataset...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
texts = [x['text'] for x in dataset["train"] if x['text'].strip()]
logger.info(f"Loaded {len(texts)} text samples")

# 2. Convert sentences to phoneme sequences
import pronouncing

def sentence_to_phonemes(sentence):
    words = sentence.lower().split()
    phonemes = []
    for word in words:
        # Get pronunciations for the word
        pronunciations = pronouncing.phones_for_word(word)
        if pronunciations:
            # Use the first pronunciation
            phonemes.append(pronunciations[0])
        else:
            # Skip words not in the dictionary
            continue
    return ' '.join(phonemes)

logger.info("Converting sentences to phonemes...")
data_pairs = []
for sent in texts:
    if len(sent.split()) < 3 or len(sent.split()) > 20:  # Filter very short or long sentences
        continue
    
    phonemes = sentence_to_phonemes(sent)
    if phonemes and phonemes.strip():
        data_pairs.append({'phonemes': phonemes, 'sentence': sent})

    if len(data_pairs) % 1000 == 0:
        logger.info(f"Processed {len(data_pairs)} sentence-phoneme pairs")

logger.info(f"Created {len(data_pairs)} valid phoneme-sentence pairs")

# 3. Create train/validation split
random.shuffle(data_pairs)
split_idx = int(len(data_pairs) * 0.95)
train_pairs = data_pairs[:split_idx]
val_pairs = data_pairs[split_idx:]

# 4. Load Llama-2-7b with 4-bit quantization
logger.info("Loading Llama-2-7b model...")
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Configure 4-bit quantization for memory efficiency
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

# 5. Apply LoRA for parameter-efficient fine-tuning
logger.info("Applying LoRA for efficient fine-tuning...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Prepare tokenization function with prompt
def preprocess(example):
    # Format with prompt as shown in the VALLR paper
    input_text = f"translate the phonemes {example['phonemes']} to a sentence:"
    target_text = example['sentence']
    
    # Tokenize input and target
    model_inputs = tokenizer(input_text, truncation=True, max_length=256, padding="max_length")
    target_tokens = tokenizer(target_text, truncation=True, max_length=256, padding="max_length")
    
    # Set up labels for causal LM training
    model_inputs["labels"] = target_tokens["input_ids"]
    return model_inputs

# 7. Tokenize datasets
logger.info("Tokenizing datasets...")
from datasets import Dataset
train_dataset = Dataset.from_list(train_pairs)
val_dataset = Dataset.from_list(val_pairs)

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./llama2_phoneme2sentence",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
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

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10. Train
logger.info("Starting training...")
trainer.train()

# 11. Save the model
logger.info("Saving model...")
model.save_pretrained("./llama2_phoneme2sentence")
tokenizer.save_pretrained("./llama2_phoneme2sentence")
logger.info("Training complete. Model saved to ./llama2_phoneme2sentence")
