# Visual-Language Projector Architecture Diagrams

This document provides visual architecture diagrams for the key projector types in the VSR-LLM codebase to complement the detailed descriptions in `projector_architectures.md`.

## Base Projector Types

### Linear and MLP Projectors

```
LinearProjector:
┌───────────┐      ┌───────────┐      ┌───────────┐
│           │      │           │      │           │
│  Visual   │─────▶│  Linear   │─────▶│  Output   │
│  Features │      │  Layer    │      │  Features │
│           │      │           │      │           │
└───────────┘      └───────────┘      └───────────┘


MLPProjector:
┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│           │      │           │      │           │      │           │      │           │
│  Visual   │─────▶│  Linear1  │─────▶│   GELU    │─────▶│  Linear2  │─────▶│  Output   │
│  Features │      │           │      │           │      │           │      │  Features │
│           │      │           │      │           │      │           │      │           │
└───────────┘      └───────────┘      └───────────┘      └───────────┘      └───────────┘
```

### QFormer Architecture

```
┌─────────────┐     ┌─────────────┐
│             │     │             │
│   Visual    │────▶│   Visual    │
│  Features   │     │ Projection  │──┐
│             │     │             │  │
└─────────────┘     └─────────────┘  │
                                     ▼
┌─────────────┐     ┌──────────────────────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │                              │     │             │     │             │
│   Query     │────▶│  Multi-Layer Cross-Attention │────▶│   Output    │────▶│   Output    │
│   Tokens    │     │        Architecture          │     │ Projection  │     │  Features   │
│ (Learnable) │     │                              │     │             │     │             │
└─────────────┘     └──────────────────────────────┘     └─────────────┘     └─────────────┘
                     │                              │
                     │   ┌──────────────────┐      │
                     │   │                  │      │
                     │   │   Self-Attention │      │
                     │   │    for Queries   │      │
                     │   │                  │      │
                     │   └────────┬─────────┘      │
                     │            │                │
                     │            ▼                │
                     │   ┌──────────────────┐      │
                     │   │                  │      │
                     │   │  Cross-Attention │      │
                     │   │  Queries→Visual  │      │
                     │   │                  │      │
                     │   └────────┬─────────┘      │
                     │            │                │
                     │            ▼                │
                     │   ┌──────────────────┐      │
                     │   │                  │      │
                     │   │   Feed-Forward   │      │
                     │   │      Network     │      │
                     │   │                  │      │
                     │   └──────────────────┘      │
                     │                              │
                     └──────────────────────────────┘
```

### CrossAttention Architecture

```
┌─────────────┐     ┌─────────────┐
│             │     │             │
│   Visual    │────▶│   Visual    │
│  Features   │     │ Projection  │──┐
│             │     │             │  │
└─────────────┘     └─────────────┘  │
                                     │
                                     ▼
┌─────────────┐     ┌─────────────────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │                         │     │             │     │             │
│   Output    │────▶│  Multiple Cross-Attn    │────▶│   Output    │────▶│   Output    │
│   Tokens    │     │        Blocks           │     │ Projection  │     │  Features   │
│ (Learnable) │     │                         │     │             │     │             │
└─────────────┘     └─────────────────────────┘     └─────────────┘     └─────────────┘
                     │                         │
                     │   ┌─────────────┐      │
                     │   │             │      │
                     │   │Self-Attention│     │
                     │   │             │      │
                     │   └──────┬──────┘      │
                     │          │             │
                     │          ▼             │
                     │   ┌─────────────┐      │
                     │   │             │      │
                     │   │Cross-Attention     │
                     │   │             │      │
                     │   └──────┬──────┘      │
                     │          │             │
                     │          ▼             │
                     │   ┌─────────────┐      │
                     │   │             │      │
                     │   │  Feed-Fwd   │      │
                     │   │             │      │
                     │   └─────────────┘      │
                     │                         │
                     └─────────────────────────┘
```

### Perceiver Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Visual   │─────┐
│ Features  │     │ Projection│     │
│           │     │           │     │
└───────────┘     └───────────┘     │
                                    │
                                    ▼
┌───────────┐     ┌───────────────────────────┐     ┌───────────┐     ┌───────────┐
│           │     │                           │     │           │     │           │
│  Latent   │────▶│  Multiple Perceiver Blocks│────▶│  Output   │────▶│  Output   │
│  Array    │     │                           │     │ Projection│     │ Features  │
│(Learnable)│     └───────────────────────────┘     │           │     │           │
└───────────┘      │                           │    └───────────┘     └───────────┘
                   │   ┌───────────────────┐   │
                   │   │                   │   │
                   │   │  Cross-Attention  │   │
                   │   │  Latents→Visual   │   │
                   │   │                   │   │
                   │   └─────────┬─────────┘   │
                   │             │             │
                   │             ▼             │
                   │   ┌───────────────────┐   │
                   │   │                   │   │
                   │   │   Self-Attention  │   │
                   │   │    for Latents    │   │
                   │   │                   │   │
                   │   └─────────┬─────────┘   │
                   │             │             │
                   │             ▼             │
                   │   ┌───────────────────┐   │
                   │   │                   │   │
                   │   │   Feed-Forward    │   │
                   │   │      Network      │   │
                   │   │                   │   │
                   │   └───────────────────┘   │
                   │                           │
                   └───────────────────────────┘
```

### AdaptiveQueryProjector Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Input    │───┐
│ Features  │     │ Projection│   │
│           │     │           │   │
└───────────┘     └───────────┘   │
                                  │
┌───────────┐                     │
│           │                     │
│  Query    │                     │
│  Tokens   │                     │
│(Learnable)│                     │
└─────┬─────┘                     │
      │                           │
      │       ┌───────────┐       │
      └──────▶│ Concatenate│◀─────┘
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐     
              │ Custom    │     
              │ Attention │     
              │ Mask      │     
              └─────┬─────┘     
                    │
                    ▼
              ┌───────────┐       ┌───────────┐       ┌───────────┐
              │Transformer │       │           │       │           │
              │Encoder with│──────▶│  Output   │──────▶│  Output   │
              │Custom Mask │       │Projection │       │ Features  │
              └───────────┘       └───────────┘       └───────────┘
```

### HierarchicalMoEProjector Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Input    │
│ Features  │     │ Projection│
│           │     │           │
└───────────┘     └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │Sparse Router
                  │ Generation │
                  └─────┬─────┘
                        │
                        ▼
┌─────────────────────────────────────────┐
│                                         │
│         Sparse Routing Block            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Expert 1 │  │ Expert 2│  │ Expert N│  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │       │
│       └────────────┼────────────┘       │
│                    │                    │
└────────────────────┼────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│                                         │
│         Sparse Routing Block            │
│                 ...                     │
│                                         │
└────────────────────┼────────────────────┘
                     │
                     ▼
               ┌───────────┐     ┌───────────┐
               │  Output   │────▶│  Output   │
               │Projection │     │ Features  │
               └───────────┘     └───────────┘
```

## Text-Aware Projector Types

### Text-Aware Cross-Attention Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Visual   │────┐
│ Features  │     │ Projection│    │
│           │     │           │    │
└───────────┘     └───────────┘    │
                                   │
┌───────────┐     ┌───────────┐    │
│  Text     │     │   BERT    │    │
│  Tokens   │────▶│ Embeddings│    │
│           │     │           │    │
└───────────┘     └─────┬─────┘    │
                        │          │
                        ▼          │
                  ┌───────────┐    │
                  │   Text    │    │
                  │ Projection│    │
                  └─────┬─────┘    │
                        │          │
┌───────────┐          │          │
│  Output   │          │          │
│  Tokens   │          │          │
│(Learnable)│          │          │
└─────┬─────┘          │          │
      │                │          │
      ▼                ▼          ▼
┌─────────────────────────────────────────┐
│          Multi-Modal Attention Layer    │
│  ┌─────────────────┐                    │
│  │  Self-Attention │                    │
│  └────────┬────────┘                    │
│           │                             │
│           ▼                             │
│  ┌─────────────────┐                    │
│  │Cross-Attention  │                    │
│  │ Queries→Visual  │                    │
│  └────────┬────────┘                    │
│           │                             │
│           ▼                             │
│  ┌─────────────────┐                    │
│  │Cross-Attention  │                    │
│  │ Queries→Text    │                    │
│  └────────┬────────┘                    │
│           │                             │
│           ▼                             │
│  ┌─────────────────┐                    │
│  │  Feed-Forward   │                    │
│  └────────┬────────┘                    │
│           │                             │
└───────────┼─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│          Multi-Modal Attention Layer    │
│                    ...                  │
└───────────┼─────────────────────────────┘
            │
            ▼
      ┌───────────┐     ┌───────────┐
      │  Output   │────▶│  Output   │
      │ Projection│     │ Features  │
      └───────────┘     └───────────┘
```

### Text-Aware Adaptive Query Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Input    │─────┐
│ Features  │     │ Projection│     │
│           │     │           │     │
└───────────┘     └───────────┘     │
                                    │
┌───────────┐     ┌───────────┐     │
│  Text     │     │   BERT    │     │
│  Tokens   │────▶│ Embeddings│     │
│           │     │           │     │
└───────────┘     └─────┬─────┘     │
                        │           │
                        ▼           │
                  ┌───────────┐     │
                  │   Text    │     │
                  │ Projection│     │
                  └─────┬─────┘     │
                        │           │
┌───────────┐          │           │
│  Query    │          │           │
│  Tokens   │          │           │
│(Learnable)│          │           │
└─────┬─────┘          │           │
      │                │           │
      ▼                ▼           │
┌─────────────────────────────┐    │
│                             │    │
│   Cross-Attention from      │    │
│   Queries to Text           │    │
│                             │    │
└─────────────┬───────────────┘    │
              │                    │
              ▼                    │
        ┌───────────┐              │
        │Conditioned│              │
        │  Queries  │              │
        └─────┬─────┘              │
              │                    │
              ▼                    ▼
        ┌───────────┐              │
        │           │              │
        │Concatenate│◀─────────────┘
        │           │
        └─────┬─────┘
              │
              ▼
        ┌───────────┐
        │ Custom    │
        │ Attention │
        │ Mask      │
        └─────┬─────┘
              │
              ▼
        ┌───────────┐     ┌───────────┐     ┌───────────┐
        │Transformer│     │           │     │           │
        │  Encoder  │────▶│  Output   │────▶│  Output   │
        │           │     │ Projection│     │ Features  │
        └───────────┘     └───────────┘     └───────────┘
```

### Text-Aware Hierarchical MoE Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Input    │─────┐
│ Features  │     │ Projection│     │
│           │     │           │     │
└───────────┘     └───────────┘     │
                                    │
┌───────────┐     ┌───────────┐     │
│  Text     │     │   BERT    │     │
│  Tokens   │────▶│ Embeddings│     │
│           │     │           │     │
└───────────┘     └─────┬─────┘     │
                        │           │
                        ▼           │
                  ┌───────────┐     │
                  │   Text    │     │
                  │ Projection│     │
                  └─────┬─────┘     │
                        │           │
                        ▼           │
                  ┌───────────┐     │
                  │Text-Based │     │
                  │Routing Bias│    │
                  └─────┬─────┘     │
                        │           │
                        │           │
                        │           │
                        ▼           ▼
┌─────────────────────────────────────────────┐
│        Text-Aware Sparse Routing Block      │
│                                             │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
│  │  Expert 1  │  │  Expert 2 │  │ Expert N │ │
│  └─────┬─────┘  └─────┬─────┘  └────┬─────┘ │
│        │              │              │      │
│        └──────────────┼──────────────┘      │
│                       │                     │
└───────────────────────┼─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│        Text-Aware Sparse Routing Block      │
│                     ...                     │
└───────────────────────┼─────────────────────┘
                        │
                        ▼
                  ┌───────────┐
                  │ Final     │
                  │ Attention │
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐     ┌───────────┐
                  │  Output   │────▶│  Output   │
                  │ Projection│     │ Features  │
                  └───────────┘     └───────────┘
```

### Text-Aware PEVL Adapter Architecture

```
┌───────────┐     ┌───────────┐
│           │     │           │
│  Visual   │────▶│  Input    │────┐
│ Features  │     │ Projection│    │
│           │     │           │    │
└───────────┘     └───────────┘    │
                                   │
┌───────────┐     ┌───────────┐    │
│  Text     │     │   BERT    │    │
│  Tokens   │────▶│ Embeddings│    │
│           │     │           │    │
└───────────┘     └─────┬─────┘    │
                        │          │
                        ▼          │
                  ┌───────────┐    │
                  │   Text    │    │
                  │ Projection│    │
                  └─────┬─────┘    │
                        │          │
                        │          │
┌───────────┐          │          │
│  Output   │          │          │
│  Tokens   │          │          │
│           │          │          │
└─────┬─────┘          │          │
      │                │          │
      │                │          │
      ▼                ▼          ▼
┌─────────────────────────────────────────┐
│      Text-Conditioned Adapter Block     │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │     Down-Projection              │    │
│  │  Conditioned by Text Features    │    │
│  └──────────────────┬──────────────┘    │
│                     │                    │
│                     ▼                    │
│  ┌─────────────────────────────────┐    │
│  │     Activation Function         │    │
│  └──────────────────┬──────────────┘    │
│                     │                    │
│                     ▼                    │
│  ┌─────────────────────────────────┐    │
│  │       Up-Projection              │    │
│  └──────────────────┬──────────────┘    │
│                     │                    │
│                     ▼                    │
│  ┌─────────────────────────────────┐    │
│  │      Residual Connection        │    │
│  └──────────────────┬──────────────┘    │
│                     │                    │
└─────────────────────┼────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│      Text-Conditioned Adapter Block     │
│                   ...                   │
└─────────────────────┼────────────────────┘
                      │
                      ▼
                ┌───────────┐
                │   Final   │
                │ Attention │
                └─────┬─────┘
                      │
                      ▼
                ┌───────────┐     ┌───────────┐
                │  Output   │────▶│  Output   │
                │ Projection│     │ Features  │
                └───────────┘     └───────────┘
```

## Full System Architecture

Below is a high-level diagram showing how these projectors fit into the overall VSR-LLM system:

```
┌─────────────────┐
│                 │
│  Visual Encoder │
│  (Frozen)       │
│                 │
└────────┬────────┘
         │
         │ Visual Features
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│    Projector    │◀─────│  Text Tokens    │ (For text-aware projectors)
│                 │      │                 │
└────────┬────────┘      └─────────────────┘
         │
         │ Projected Features
         │
         ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌─────────────┐   ┌───────────┐   ┌──────────┐ │
│  │ Instruction │   │ Projected │   │  Label   │ │
│  │ Embeddings  │ + │ Features  │ + │ Embeddings│ │
│  └─────────────┘   └───────────┘   └──────────┘ │
│                                                 │
│               Language Model                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

These diagrams illustrate the flow of data through each projector architecture and how the different components interact. The final system architecture shows how the projectors serve as a bridge between the visual encoder and the language model, with optional text conditioning for the text-aware variants. 