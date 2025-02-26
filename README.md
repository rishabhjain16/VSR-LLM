# VSP-LLM (Visual Speech Processing incorporated with LLMs)


## Introduction

WORK IN PROGRESS
This code is Forked from: https://github.com/Sally-SH/VSP-LLM
We are working to update it for specific usecase of using Avhubert for Visucal Speech Recognition
## Model checkpoint

You can find checkpoint of our model in [here](https://drive.google.com/drive/folders/1aBnm8XOWlRAGjPwcK2mYEGd8insNCx13?usp=sharing).
Move the checkpoint to [`checkpoints`](checkpoints/).

## Preparation

```
conda create -n vsr-llm python=3.9 -y
conda activate vsr-llm
git clone https://github.com/rishabhjain16/VSR-LLM.git
cd VSR-LLM
(If your pip version > 24.1, please run "pip install --upgrade pip==24.0")
pip install -r requirements.txt
cd fairseq
pip install --editable ./
pip install pip==24.0 
pip install hydra-core==1.0.7 
pip install omegaconf==2.0.4 
pip install numpy==1.23.0
pip install -U bitsandbytes
pip install protobuf==3.20
 
```

- Download AV-HuBERT pre-trained model `AV-HuBERT Large (LSR3 + VoxCeleb2)` from [here](http://facebookresearch.github.io/av_hubert).
- Download LLaMA2-7B from [here](https://huggingface.co/meta-llama/Llama-2-7b-hf).

Move the AV-HuBERT pre-trained model checkpoint and the LLaMA2-7B checkpoint to [`checkpoints`](checkpoints/).

## Data preprocessing
Follow [Auto-AVSR preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation) to preprocess the LRS3 dataset.\
Then, follow [AV-HuBERT preparation](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation) from step 3 to create manifest of LRS3 dataset.

### Generate visual speech unit and cluster counts file
Follow the steps in [`clustering`](src/clustering/) to create:
- `{train,valid}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 25Hz for AV-HuBERT features by default.

### Dataset layout

    .
    ├── lrs3
    │     ├── lrs3_video_seg24s               # Preprocessed video and audio data
    │     └── lrs3_text_seg24s                # Preprocessed text data
    ├── muavic_dataset                        # Mix of VSR data and VST(En-X) data
    │     ├── train.tsv                       # List of audio and video path for training
    │     ├── train.wrd                       # List of target label for training
    │     ├── train.cluster_counts            # List of clusters to deduplicate speech units in training
    │     ├── test.tsv                        # List of audio and video path for testing
    │     ├── test.wrd                        # List of target label for testing
    │     └── test.cluster_counts             # List of clusters to deduplicate speech units in testing
    └── test_data
          ├── vsr
          │    └── en
          │        ├── test.tsv 
          │        ├── test.wrd  
          │        └── test.cluster_counts           
          └── vst
               └── en
                   ├── es
                   :   ├── test.tsv
                   :   ├── test.wrd 
                   :   └── test.cluster_counts
                   └── pt
                       ├── test.tsv
                       ├── test.wrd 
                       └── test.cluster_counts

### Test data
The test manifest is provided in [`labels`](labels/). You need to replace the path of the LRS3 in the manifest file with your preprocessed LRS3 dataset path using the following command:
```bash
cd src/dataset
python replace_path.py --lrs3 /path/to/lrs3
```
Then modified test amanifest is saved in [`dataset`](src/dataset/)

## Training

Open the training script ([`scripts/train.sh`](https://github.com/Sally-SH/VSP-LLM/blob/main/scripts/train.sh)) and replace these variables:

```bash
# path to train dataset dir
DATA_PATH=???

# path where output trained models will be located
OUT_PATH=???
```

Run the training script:

```bash
$ bash scripts/train.sh
```

## Decoding

Open the decoding script ([`scripts/decode.sh`](https://github.com/Sally-SH/VSP-LLM/blob/main/scripts/decode.sh)) and replace these variables:

```bash
# language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)
LANG=???

# path to the trained model
MODEL_PATH=???

# path where decoding results and scores will be located
OUT_PATH=???
```

Run the decoding script:

```bash
$ bash scripts/decode.sh
```
