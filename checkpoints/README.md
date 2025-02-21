# Checkpoints for train and inference
Please move the following checkpoints to here.
- AV-HuBERT Large (LSR3 + VoxCeleb2) checkpoint
- LLaMA2-7B checkpoint
- VSP-LLM checkpoints (checkpoint_freeze.pt, checkpoint_finetune.pt)


# To download models from Hugging Face:

1. pip install -U "huggingface_hub[cli]" 
2. git config --global credential.helper store 
3. huggingface-cli login  #---> Enter your token from Hugging face (if required)
4. huggingface-cli download model_name --local-dir ./directory_name
