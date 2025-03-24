# VSR-LLM System Architecture

This document provides a comprehensive overview of the complete VSR-LLM system architecture, explaining how projectors interface with the Language Model, how data flows through the system, and how training/inference works.

## Table of Contents
1. [Overall System Architecture](#overall-system-architecture)
2. [Data Flow](#data-flow)
3. [Training Pipeline](#training-pipeline)
4. [Loss Calculation](#loss-calculation)
5. [Inference Pipeline](#inference-pipeline)
6. [Integration with LLMs](#integration-with-llms)

## Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                     VSR-LLM System                                        │
│                                                                                          │
│  ┌────────────┐      ┌──────────────┐      ┌────────────────────────────────────────┐    │
│  │            │      │              │      │                                        │    │
│  │   Input    │      │    Visual    │      │              Projector                 │    │
│  │  (Images/  │─────▶│   Encoder    │─────▶│ (Base or Text-Aware versions from      │    │
│  │   Video)   │      │   (Frozen)   │      │  projector catalog)                    │    │
│  │            │      │              │      │                                        │    │
│  └────────────┘      └──────────────┘      └───────────────────┬────────────────────┘    │
│                                                                │                         │
│                      ┌──────────────┐                          │                         │
│                      │              │                          │                         │
│                      │    Text      │                          │                         │
│                      │   Tokens     │─────────────────────────▶│                         │
│                      │(Instructions)│                          │                         │
│                      └──────────────┘                          │                         │
│                                                                │                         │
│                                                                ▼                         │
│  ┌────────────────────────────────────────────────────────────────────────────────┐     │
│  │                                                                                │     │
│  │                               Language Model                                   │     │
│  │                                                                                │     │
│  │  ┌─────────────┐   ┌─────────────────┐   ┌───────────────┐                    │     │
│  │  │             │   │                 │   │               │                    │     │
│  │  │ Instruction │   │   Projected     │   │    Label      │                    │     │
│  │  │ Embeddings  │ + │   Features      │ + │  Embeddings   │                    │     │
│  │  │             │   │                 │   │               │                    │     │
│  │  └─────────────┘   └─────────────────┘   └───────────────┘                    │     │
│  │                                          ▲                                    │     │
│  │                                          │                                    │     │
│  └─────────────────────────────────────────┬┼────────────────────────────────────┘     │
│                                            ││                                           │
│  ┌────────────────┐                        ││                                           │
│  │                │                        ││                                           │
│  │  Ground Truth  │────────────────────────┘│                                           │
│  │  Text Labels   │                         │                                           │
│  │                │                         │                                           │
│  └────────────────┘                         │                                           │
│                                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐                        │
│  │                     Loss Calculation                         │                        │
│  │                                                             │                        │
│  │  ┌───────────────────────────────┐   ┌──────────────────┐   │                        │
│  │  │ Language Modeling Loss        │   │ Auxiliary Losses │   │                        │
│  │  │ (Compare LLM outputs with     │ + │ (Contrastive,    │   │                        │
│  │  │  ground truth text labels)    │   │  Reconstruction) │   │                        │
│  │  └───────────────────────────────┘   └──────────────────┘   │                        │
│  │                                                             │                        │
│  └─────────────────────────────────────────────────────────────┘                        │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

The VSR-LLM system processes data in the following sequence:

1. **Input Processing**:
   - **Visual Input**: Images or video frames are processed through a frozen visual encoder to extract features.
   - **Text Input (Instructions)**: Instruction text is tokenized and processed for text-aware projectors.
   - **Text Labels**: Ground truth captions or descriptions are processed for training.

2. **Feature Projection**:
   - The visual features are passed through the selected projector (from the various options in the projector catalog).
   - For text-aware projectors, text instruction tokens are also passed to guide the projection process.
   - The projector transforms the visual features into a format compatible with the LLM embedding space.

3. **Language Model Integration**:
   - The projected features are combined with instruction embeddings and label embeddings.
   - This combined input is passed to the language model for processing.

4. **Output Generation and Loss Calculation**:
   - During training, the model produces logits which are compared against ground truth text labels.
   - During inference, the language model generates text output based on the visual inputs.

## Training Pipeline with Label Processing

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│                      │     │                      │     │                      │
│    Data Loading      │────▶│   Feature Extraction │────▶│  Projector Forward   │
│   (Images + Text     │     │   (Visual + Text)    │     │                      │
│    Labels)           │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                     │
        ┌─────────────────────────────────────────────────┐          │
        │                                                 │          │
        │  Text Label Processing                          │          │
        │  ┌─────────────┐     ┌─────────────┐           │          │
        │  │ Tokenization│────▶│ Embedding   │           │          │
        │  └─────────────┘     └──────┬──────┘           │          │
        │                             │                  │          │
        └─────────────────────────────┼──────────────────┘          │
                                      │                             │
                                      ▼                             ▼
                             ┌──────────────────────┐     ┌──────────────────────┐
                             │                      │     │                      │
                             │  Create Target       │     │    LLM Forward       │
                             │  Label Sequence      │     │                      │
                             │                      │     │                      │
                             └──────────┬───────────┘     └──────────┬───────────┘
                                        │                            │
                                        └────────────┐   ┌───────────┘
                                                     ▼   ▼
                                           ┌──────────────────────┐     ┌──────────────────────┐
                                           │                      │     │                      │
                                           │   Loss Calculation   │────▶│   Parameter Update   │
                                           │                      │     │                      │
                                           └──────────────────────┘     └──────────────────────┘
```

The training pipeline with label processing consists of:

1. **Data Loading**:
   - Loads image-text pairs from the dataset
   - Each pair includes the image and corresponding ground truth text label/caption
   - Applies appropriate preprocessing and augmentation
   - Prepares instruction templates

2. **Feature Extraction**:
   - Passes images through the frozen visual encoder
   - Tokenizes and processes instruction text
   - Applies any necessary normalization

3. **Text Label Processing**:
   - Tokenizes the ground truth text labels
   - Converts tokens to embeddings for LLM input
   - Creates target label sequence for loss calculation

4. **Projector Forward Pass**:
   - Runs the visual features through the selected projector
   - For text-aware projectors, incorporates instruction text
   - Produces projected features aligned with LLM embedding space

5. **LLM Forward Pass**:
   - Combines instruction embeddings, projected features, and label embeddings
   - Passes this combined input through the LLM
   - Produces logits for next-token prediction

6. **Loss Calculation**:
   - Computes language modeling loss by comparing logits with ground truth text labels
   - May include additional losses like contrastive or auxiliary losses
   - Combines losses according to weighting scheme

7. **Parameter Update**:
   - Computes gradients through backpropagation
   - Only updates parameters in the projector while keeping LLM frozen
   - Applies optimization step with learning rate scheduling

## Loss Calculation

### Primary Loss: Language Modeling with Text Labels

The main loss is the language modeling (LM) loss, calculated as:

```
LM Loss = CrossEntropyLoss(logits, ground_truth_text_labels)
```

Where:
- `logits` are the model's predictions (output from the LLM)
- `ground_truth_text_labels` are the target tokens from the dataset text captions/descriptions

This process works as follows:

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │     │                │
│ Ground Truth  │────▶│ Tokenization   │────▶│ Create Target  │────▶│ Compare with   │
│ Text Labels   │     │                │     │ Label Sequence │     │ Model Output   │
│                │     │                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
                                                                             │
                                                                             │
                                                                             ▼
                                                                     ┌────────────────┐
                                                                     │                │
                                                                     │ Cross-Entropy  │
                                                                     │ Loss           │
                                                                     │                │
                                                                     └────────────────┘
```

For example, if the ground truth text label is "A cat sitting on a mat":
1. It's tokenized into token IDs
2. A target label sequence is created for autoregressive prediction
3. The LLM predicts the next token at each position
4. Cross-entropy loss compares each prediction against the true next token

### Optional Auxiliary Losses

For some projector types, auxiliary losses may be used:

1. **Contrastive Loss** (for MultiScaleContrastiveProjector):
   ```
   Contrastive Loss = -log(exp(sim(p, p+)/τ) / Σ exp(sim(p, p')/τ))
   ```
   Where:
   - `p` is the projection of an image
   - `p+` is a positive example (same image at different scale)
   - `p'` are other examples in the batch
   - `τ` is temperature parameter
   - `sim` is cosine similarity

2. **Reconstruction Loss** (for some projectors):
   ```
   Reconstruction Loss = MSE(reconstructed_features, original_features)
   ```

3. **Routing Loss** (for MoE-based projectors):
   ```
   Routing Loss = CV(router_probabilities) + Load_Balancing_Loss
   ```
   Where CV is the coefficient of variation to encourage balanced expert utilization.

The total loss is a weighted combination:
```
Total Loss = α * LM_Loss + β * Contrastive_Loss + γ * Auxiliary_Losses
```
Where α, β, and γ are weighting hyperparameters.

## Inference Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│            │     │            │     │            │     │            │     │            │
│  Image/    │────▶│  Visual    │────▶│ Projector  │────▶│    LLM     │────▶│  Text      │
│  Video     │     │  Encoder   │     │ Processing │     │ Generation │     │  Output    │
│            │     │            │     │            │     │            │     │            │
└────────────┘     └────────────┘     └────────────┘     └────────────┘     └────────────┘
```

During inference:
1. The system takes an image/video input
2. Extracts visual features with the frozen encoder
3. Processes them through the selected projector
4. Feeds the projected features to the LLM along with instructions
5. The LLM generates text output in response to the visual content

## Integration with LLMs

### Detailed Process Flow

```
                                         ┌──────────────────────────────────┐
                                         │           Instruction            │
                                         │           Text Template          │
                                         └────────────────┬─────────────────┘
                                                         │
                                                         ▼
                                         ┌──────────────────────────────────┐
                                         │                                  │
                                         │         Tokenization             │
                                         │                                  │
                                         └────────────────┬─────────────────┘
                                                         │
┌──────────────────────────────┐                         │
│                              │                         │
│      Projected Visual        │                         │
│        Features              │                         │
│                              │                         │
└──────────────┬───────────────┘                         │
               │                                         │
               └───────────────┐         ┌───────────────┘
                               ▼         ▼
                       ┌──────────────────────────┐     ┌────────────────────┐
                       │                          │     │                    │
                       │  Create input sequence   │────▶│  LLM Forward Pass  │
                       │                          │     │                    │
                       └──────────────────────────┘     └──────────┬─────────┘
                                 ▲                                 │
                                 │                                 │
                ┌────────────────┴─────────┐                      │
                │                          │                      │
                │  Text Labels             │                      │
                │  (during training)       │                      │
                │                          │                      │
                └──────────────────────────┘                      │
                                                                  ▼
                                                       ┌────────────────────┐
                                                       │                    │
                                                       │    Decode Output   │
                                                       │                    │
                                                       └────────────────────┘
```

### Input Sequence Formation

The input to the LLM is created by concatenating:

1. **Instruction Embeddings**: Text embeddings from a template like "Describe what you see in this image:"
2. **Projected Features**: The output from the projector, representing visual content
3. **Label Embeddings** (during training): Ground truth text embeddings from the dataset

The sequence is formatted as:
```
[Instruction Embeddings] + [Projected Visual Features] + [Label Embeddings (during training)]
```

### LLM Processing

During forward pass:
1. The LLM processes the combined input sequence
2. The model predicts the next tokens in the sequence
3. During training, these predictions are compared to ground truth text labels
4. During inference, the predictions are used for text generation

The LLM itself is typically kept frozen during training, with only the projector parameters being updated. This efficient approach allows adaptation of the visual features to the LLM's embedding space without modifying the language model itself.

## Text-Aware vs. Base Projectors

### Base Projector Workflow
```
┌────────────┐     ┌────────────┐     ┌────────────┐
│            │     │            │     │            │
│  Visual    │────▶│ Projector  │────▶│ Projection │
│  Features  │     │ (Base)     │     │ Output     │
│            │     │            │     │            │
└────────────┘     └────────────┘     └────────────┘
```

### Text-Aware Projector Workflow
```
┌────────────┐     ┌────────────┐
│            │     │            │
│  Visual    │────▶│            │
│  Features  │     │            │     ┌────────────┐
│            │     │ Text-Aware │────▶│            │
└────────────┘     │ Projector  │     │ Projection │
                   │            │     │ Output     │
┌────────────┐     │            │     │            │
│            │     │            │     │            │
│  Text      │────▶│            │     │            │
│  Tokens    │     │            │     │            │
│            │     │            │     │            │
└────────────┘     └────────────┘     └────────────┘
```

The key difference is that text-aware projectors integrate instruction text directly during the projection process, enabling more fine-grained control over visual feature extraction based on textual context. 

## Complete Label Processing Flow

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Training Data Sample                                      │
│                                                                                    │
│  ┌───────────────┐            ┌──────────────────────┐                             │
│  │               │            │                      │                             │
│  │    Image      │            │   Ground Truth Text  │                             │
│  │               │            │   "A cat on a mat"   │                             │
│  │               │            │                      │                             │
│  └───────┬───────┘            └──────────┬───────────┘                             │
│          │                               │                                         │
└──────────┼───────────────────────────────┼─────────────────────────────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────┐            ┌──────────────────────┐
│                  │            │                      │
│ Visual Encoder   │            │    Tokenization      │
│                  │            │                      │
└──────────┬───────┘            └──────────┬───────────┘
           │                               │
           ▼                               ▼
┌──────────────────┐            ┌──────────────────────┐
│                  │            │                      │
│ Visual Features  │            │  Token IDs: [101,    │
│                  │            │  2023, 2006, 1037,   │
│                  │            │  17985, 102]         │
└──────────┬───────┘            └──────────┬───────────┘
           │                               │
           ▼                               │
┌──────────────────┐                       │
│                  │                       │
│    Projector     │                       │
│                  │                       │
└──────────┬───────┘                       │
           │                               │
           ▼                               ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                    │
│                           Language Model Processing                                │
│                                                                                    │
│   ┌─────────────┐     ┌───────────────┐      ┌───────────────┐                    │
│   │ Instruction │     │  Projected    │      │  Label        │                    │
│   │ Embeddings  │ +   │  Features     │  +   │  Embeddings   │                    │
│   └─────────────┘     └───────────────┘      └───────────────┘                    │
│                                                                                    │
│                                   │                                                │
│                                   ▼                                                │
│                      ┌────────────────────────┐                                    │
│                      │                        │                                    │
│                      │     Model Outputs      │                                    │
│                      │                        │                                    │
│                      └────────────┬───────────┘                                    │
│                                   │                                                │
└───────────────────────────────────┼────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │                        │
                        │  Compare with Ground   │
                        │  Truth Token IDs       │
                        │                        │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │                        │
                        │  Language Modeling     │
                        │  Loss                  │
                        │                        │
                        └────────────────────────┘
``` 