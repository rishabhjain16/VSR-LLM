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
│                      │              │                          │                         │
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
│  │                                                                                │     │
│  │                                                                                │     │
│  └───────────────────────────────────────┬────────────────────────────────────────┘     │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐                        │
│  │                     Loss Calculation                         │                        │
│  │                                                             │                        │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │                        │
│  │  │ Language     │   │ Contrastive  │   │ Auxiliary    │     │                        │
│  │  │ Modeling Loss│ + │ Loss (opt.)  │ + │ Losses (opt.)│     │                        │
│  │  └──────────────┘   └──────────────┘   └──────────────┘     │                        │
│  │                                                             │                        │
│  └─────────────────────────────────────────────────────────────┘                        │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

The VSR-LLM system processes data in the following sequence:

1. **Input Processing**:
   - **Visual Input**: Images or video frames are processed through a frozen visual encoder to extract features.
   - **Text Input**: Instruction text is tokenized and processed for text-aware projectors.

2. **Feature Projection**:
   - The visual features are passed through the selected projector (from the various options in the projector catalog).
   - For text-aware projectors, text tokens are also passed to guide the projection process.
   - The projector transforms the visual features into a format compatible with the LLM embedding space.

3. **Language Model Integration**:
   - The projected features are combined with instruction embeddings and label embeddings.
   - This combined input is passed to the language model for processing.

4. **Output Generation**:
   - During inference, the language model generates text output based on the visual inputs.
   - During training, the model produces logits for loss calculation.

## Training Pipeline

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│                      │     │                      │     │                      │
│    Data Loading      │────▶│   Feature Extraction │────▶│  Projector Forward   │
│                      │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                     │
                                                                     ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│                      │     │                      │     │                      │
│   Parameter Update   │◀────│   Loss Calculation   │◀────│    LLM Forward       │
│                      │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

The training pipeline consists of:

1. **Data Loading**:
   - Loads image-text pairs from the dataset
   - Applies appropriate preprocessing and augmentation
   - Prepares instruction templates and label formats

2. **Feature Extraction**:
   - Passes images through the frozen visual encoder
   - Tokenizes and processes instruction text
   - Applies any necessary normalization

3. **Projector Forward Pass**:
   - Runs the visual features through the selected projector
   - For text-aware projectors, incorporates instruction text
   - Produces projected features aligned with LLM embedding space

4. **LLM Forward Pass**:
   - Combines instruction embeddings, projected features, and label embeddings
   - Passes this combined input through the LLM
   - Produces logits for next-token prediction

5. **Loss Calculation**:
   - Computes language modeling loss (cross-entropy)
   - May include additional losses like contrastive or auxiliary losses
   - Combines losses according to weighting scheme

6. **Parameter Update**:
   - Computes gradients through backpropagation
   - Only updates parameters in the projector while keeping LLM frozen
   - Applies optimization step with learning rate scheduling

## Loss Calculation

### Primary Loss: Language Modeling

The main loss is the language modeling (LM) loss, calculated as:

```
LM Loss = CrossEntropyLoss(logits, target_labels)
```

Where:
- `logits` are the model's predictions (output from the LLM)
- `target_labels` are the ground truth next tokens in the sequence

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
                                                                   │
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
3. **Label Embeddings** (during training): Ground truth text embeddings

The sequence is formatted as:
```
[Instruction Embeddings] + [Projected Visual Features] + [Label Embeddings (during training)]
```

### LLM Processing

During forward pass:
1. The LLM processes the combined input sequence
2. The model predicts the next tokens in the sequence
3. During training, these predictions are compared to ground truth
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