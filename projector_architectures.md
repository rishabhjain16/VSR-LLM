# Vision-Language Projector Architectures

This document provides a comprehensive overview of all the projector architectures implemented in the VSR-LLM codebase. Each projector is responsible for transforming visual features into a format suitable for integration with large language models.

## Table of Contents

1. [Base Projectors](#base-projectors)
   - [LinearProjector](#linearprojector)
   - [MLPProjector](#mlpprojector)
   - [QFormerProjector](#qformerprojector)
   - [CrossAttentionProjector](#crossattentionprojector)
   - [PerceiverProjector](#perceiverprojector)
   - [AdaptiveQueryProjector](#adaptivequeryprojector)
   - [FusionRefinementProjector](#fusionrefinementprojector)
   - [GatedCrossAttentionProjector](#gatedcrossattentionprojector)
   - [HierarchicalMoEProjector](#hierarchicalmoeprojector)
   - [EnhancedQFormerProjector](#enhancedqformerprojector)
   - [MultiScaleContrastiveProjector](#multiscalecontrastiveprojector)
   - [PEVLAdapter](#pevladapter)
   - [SelfAggregatingLinearProjector](#selfaggregatinglinearprojector)
   - [SelfAggregatingMLPProjector](#selfaggregatingmlpprojector)
   - [BLIP2QFormer](#blip2qformer)

2. [Text-Aware Projectors](#text-aware-projectors)
   - [TextAwareCrossAttentionProjector](#textawarecrossattentionprojector)
   - [TextAwarePerceiverProjector](#textawareperceiverprojector)
   - [TextAwareAdaptiveQueryProjector](#textawareadaptivequeryprojector)
   - [TextAwareHierarchicalMoEProjector](#textawarehierarchicalmoeprojector)
   - [TextAwareFusionRefinementProjector](#textawarefusionrefinementprojector)
   - [TextAwareGatedCrossAttentionProjector](#textawaregatedcrossattentionprojector)
   - [TextAwareMultiScaleContrastiveProjector](#textawaremultiscalecontrastiveprojector)
   - [TextAwarePEVLAdapter](#textawarepevladapter)

---

## Base Projectors

### LinearProjector

**Architecture**: The simplest projector, consisting of a single linear layer.

**Functionality**: Performs a direct linear transformation from the input dimension to the output dimension.

```
input → Linear(input_dim, output_dim) → output
```

**Inspired by**: Standard linear projections used in many vision-language models for feature transformation.

**References**:
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

---

### MLPProjector

**Architecture**: A multi-layer perceptron with configurable depth, hidden dimensions, and activation functions.

**Functionality**: Processes input features through multiple linear layers with non-linear activations to capture more complex feature representations.

```
input → Linear → Activation → [Linear → Activation]... → Linear → output
```

**Inspired by**: MLP-based projection heads from contrastive learning models.

**References**:
- Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

---

### QFormerProjector

**Architecture**: Implements a Query Transformer architecture with learnable query tokens that attend to visual features through cross-attention.

**Functionality**: 
1. Initializes a set of learnable query tokens
2. Projects input visual features to a hidden dimension
3. Processes the queries through multiple cross-attention layers, allowing them to attend to visual features
4. Projects the updated queries to the output dimension

```
Visual Features → Visual Projection
Query Tokens → Multi-layer Cross-Attention with Visual Features → Output Projection
```

**Inspired by**: The Query Transformer architecture from BLIP and BLIP-2.

**References**:
- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML 2022.
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.

---

### CrossAttentionProjector

**Architecture**: Uses multiple cross-attention blocks to process visual features.

**Functionality**:
1. Projects input features to a hidden dimension
2. Initializes a set of learnable query tokens
3. Processes through multiple cross-attention blocks where:
   - Queries attend to themselves (self-attention)
   - Queries attend to visual features (cross-attention)
4. Projects the processed queries to the output dimension

```
Visual Features → Visual Projection
Query Tokens → [Self-Attention → Cross-Attention with Visual]... → Output Projection
```

**Inspired by**: Attention mechanisms from Transformers and cross-attention from multimodal models.

**References**:
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS 2017.
- Alayrac, J.B., et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. NeurIPS 2022.

---

### PerceiverProjector

**Architecture**: Based on the Perceiver architecture, using a small set of latent vectors that iteratively attend to visual features.

**Functionality**:
1. Projects input visual features
2. Initializes a set of learnable latent vectors
3. Iteratively updates the latents through perceiver blocks:
   - Cross-attention: latents attend to visual features
   - Self-attention: latents attend to themselves
4. Projects the final latents to the output dimension

```
Visual Features → Visual Projection
Latent Vectors → [Cross-Attention with Visual → Self-Attention]... → Output Projection
```

**Inspired by**: Google DeepMind's Perceiver architecture.

**References**:
- Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. ICML 2021.
- Jaegle, A., et al. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. ICLR 2022.

---

### AdaptiveQueryProjector

**Architecture**: Uses learnable query tokens with a transformer encoder to process visual features, with a specially designed attention mask.

**Functionality**:
1. Projects input features to hidden dimension
2. Initializes learnable query tokens
3. Concatenates queries and projected features
4. Creates a custom attention mask where:
   - Queries can attend to all tokens
   - Input features can only attend to themselves
5. Processes through a transformer encoder with this mask
6. Extracts only the query token outputs
7. Projects to the output dimension

```
Visual Features → Visual Projection
Query Tokens + Projected Features → Transformer with Custom Mask → Extract Queries → Output Projection
```

**Inspired by**: Combines elements from DETR and modern vision-language models.

**References**:
- Carion, N., et al. (2020). End-to-End Object Detection with Transformers. ECCV 2020.
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.

---

### FusionRefinementProjector

**Architecture**: Implements a multi-stage refinement process for visual features with progressive fusion.

**Functionality**:
1. Projects input features to a hidden dimension
2. Processes through multiple refinement stages, where each stage:
   - Applies self-attention to current features
   - Fuses with outputs from previous stages using cross-attention
   - Applies a feed-forward network
3. Projects the final refined features to the output dimension

```
Visual Features → Visual Projection → [Refinement Stage 1 → ... → Refinement Stage N] → Output Projection
```

**Inspired by**: Progressive refinement approaches in computer vision and NLP.

**References**:
- Carion, N., et al. (2020). End-to-End Object Detection with Transformers. ECCV 2020.
- Zhu, X., et al. (2020). Deformable DETR: Deformable Transformers for End-to-End Object Detection. ICLR 2021.

---

### GatedCrossAttentionProjector

**Architecture**: Enhances cross-attention with a gating mechanism to control information flow.

**Functionality**:
1. Projects input features to a hidden dimension
2. Initializes learnable query tokens
3. Processes through multiple gated cross-attention blocks:
   - Cross-attention: queries attend to visual features
   - Gating mechanism decides how much new information to incorporate
4. Projects the final queries to the output dimension

```
Visual Features → Visual Projection
Query Tokens → [Gated Cross-Attention Block]... → Output Projection
```

**Inspired by**: Gating mechanisms in LSTMs and Transformers.

**References**:
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL 2019.
- Liu, Y., et al. (2022). Gated Attention as Model Integration for Vision and Language Understanding. ArXiv.

---

### HierarchicalMoEProjector

**Architecture**: Employs a Mixture-of-Experts approach with hierarchical routing for efficient feature projection.

**Functionality**:
1. Projects input features to a hidden dimension
2. Processes through multiple layers of sparse routing blocks:
   - Each block computes routing probabilities for experts
   - Routes input to top-k experts
   - Combines expert outputs weighted by routing probabilities
3. Projects the final output to the target dimension

```
Visual Features → Input Projection → [Sparse Routing Block]... → Output Projection
```

**Inspired by**: Mixture-of-Experts models from Sparsely-Gated MoE and Switch Transformers.

**References**:
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
- Fedus, W., et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR 2022.

---

### EnhancedQFormerProjector

**Architecture**: An improved version of the QFormerProjector with deeper transformers and larger feed-forward networks.

**Functionality**: Similar to QFormerProjector but with:
- Deeper transformer layers
- Larger intermediate dimensions in feed-forward networks
- Enhanced initialization procedures

```
Visual Features → Visual Projection
Query Tokens → Deeper Cross-Attention with Visual Features → Output Projection
```

**Inspired by**: BLIP-2 and improved transformer architectures.

**References**:
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.
- Liu, Y., et al. (2023). Visual Instruction Tuning. NeurIPS 2023.

---

### MultiScaleContrastiveProjector

**Architecture**: Processes visual features at multiple scales with contrastive objectives.

**Functionality**:
1. Projects input features to a hidden dimension
2. Processes features through multiple scale-specific projections
3. Applies attention pooling at each scale
4. Computes contrastive losses between scales during training
5. Combines multi-scale representations for the final output

```
Visual Features → Input Projection → [Scale-Specific Projections] → Attention Pooling → Output Projection
```

**Inspired by**: Multi-scale processing in computer vision and contrastive learning.

**References**:
- Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV 2021.

---

### PEVLAdapter

**Architecture**: Implements Parameter-Efficient Visual Learner adapters with low-rank transformations.

**Functionality**:
1. Projects input features to a hidden dimension
2. Applies task-specific bottleneck adapters:
   - Down-projects to a bottleneck dimension
   - Up-projects back to the original dimension
   - Uses residual connections
3. Projects the adapted features to the output dimension

```
Visual Features → Input Projection → [Low-Rank Adapter Layers]... → Output Projection
```

**Inspired by**: Parameter-efficient fine-tuning methods like adapters and LoRA.

**References**:
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. ICML 2019.
- Hu, E.J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Jia, M., et al. (2022). Visual Prompt Tuning. ECCV 2022.

---

### SelfAggregatingLinearProjector

**Architecture**: A simple projector that creates and self-aggregates output tokens through attention.

**Functionality**:
1. Projects input features to a hidden dimension
2. Initializes a set of output tokens
3. Computes attention scores between output tokens and projected features
4. Uses attention weights to aggregate features into output tokens
5. Projects the output tokens to the target dimension

```
Visual Features → Input Projection
Output Tokens → Cross-Attention with Features → Output Projection
```

**Inspired by**: Attention pooling mechanisms in vision-language models.

**References**:
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- Wang, W., et al. (2022). Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. CVPR 2022.

---

### SelfAggregatingMLPProjector

**Architecture**: Combines MLP processing with self-aggregating output tokens.

**Functionality**:
1. Projects input features through an MLP
2. Initializes a set of output tokens
3. Computes attention scores between output tokens and processed features
4. Uses attention weights to aggregate features into output tokens
5. Projects the output tokens to the target dimension

```
Visual Features → MLP Layers
Output Tokens → Cross-Attention with MLP Features → Output Projection
```

**Inspired by**: MLP-based feature processing with attention pooling.

**References**:
- Tolstikhin, I., et al. (2021). MLP-Mixer: An all-MLP Architecture for Vision. NeurIPS 2021.
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.

---

### BLIP2QFormer

**Architecture**: A faithful implementation of the QFormer architecture from BLIP-2.

**Functionality**:
1. Initializes learnable query tokens
2. Projects visual features
3. Processes query tokens through a BERT-like architecture with:
   - Self-attention among queries
   - Cross-attention to visual features
   - Optional cross-attention to text
4. Outputs the processed query tokens

```
Visual Features → Visual Projection
Query Tokens → [Self-Attention → Cross-Attention with Visual → (Optional) Cross-Attention with Text]... → Output
```

**Inspired by**: BLIP-2's vision-language bridging architecture.

**References**:
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.

---

## Text-Aware Projectors

Text-aware projectors extend base projectors by incorporating instruction text to guide the visual feature projection process. These projectors can better align visual representations with language models by taking into account textual instructions.

### TextAwareCrossAttentionProjector

**Architecture**: Extends CrossAttentionProjector to incorporate textual guidance through BERT embeddings.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes output tokens
4. Multi-layer processing where each layer:
   - Applies self-attention to output tokens
   - Applies cross-attention from output tokens to visual features
   - Applies cross-attention from output tokens to text features
5. Projects the final output tokens to the target dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection
Output Tokens → [Self-Attention → Cross-Attention with Visual → Cross-Attention with Text]... → Output Projection
```

**Inspired by**: Cross-attention mechanisms in multimodal transformers and BLIP-2.

**References**:
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.
- Alayrac, J.B., et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. NeurIPS 2022.

---

### TextAwarePerceiverProjector

**Architecture**: Extends PerceiverProjector to incorporate instruction text for guiding latent processing.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes latent tokens
4. Multi-layer processing where each layer:
   - Applies self-attention to latent tokens
   - Applies cross-attention from latents to visual features
   - Applies cross-attention from latents to text features
5. Projects the final latents to the output dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection
Latent Tokens → [Self-Attention → Cross-Attention with Visual → Cross-Attention with Text]... → Output Projection
```

**Inspired by**: Perceiver architecture and multimodal conditioning techniques.

**References**:
- Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. ICML 2021.
- Jaegle, A., et al. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. ICLR 2022.
- Alayrac, J.B., et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. NeurIPS 2022.

---

### TextAwareAdaptiveQueryProjector

**Architecture**: Enhances AdaptiveQueryProjector by conditioning query tokens on instruction text.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes query tokens
4. Applies cross-attention from queries to text to condition them
5. Concatenates conditioned queries with visual features
6. Processes through a transformer with a custom attention mask
7. Extracts query outputs and projects to the target dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection
Query Tokens → Cross-Attention with Text → Concatenate with Visual → Transformer with Custom Mask → Output Projection
```

**Inspired by**: Multimodal conditioning techniques in vision-language models.

**References**:
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.
- Liu, Y., et al. (2023). Visual Instruction Tuning. NeurIPS 2023.

---

### TextAwareHierarchicalMoEProjector

**Architecture**: Extends HierarchicalMoEProjector to use text for guiding expert routing.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Generates text-based routing biases for experts
4. Processes through multiple layers of text-aware sparse routing blocks:
   - Computes routing probabilities with text-conditioning
   - Routes input to top-k experts with text-informed decisions
   - Combines expert outputs weighted by routing probabilities
5. Applies final attention to aggregate features
6. Projects to the output dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection → Router Bias Generator
Input → [Text-Aware Sparse Routing Block]... → Output Attention → Output Projection
```

**Inspired by**: Mixture-of-Experts with language-conditional routing.

**References**:
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
- Mustafa, B., et al. (2022). MultiModal-MoE: Learning Multi-Modal Mixture-of-Experts through Modal-Specific Routing. NeurIPS 2022.

---

### TextAwareFusionRefinementProjector

**Architecture**: Enhances FusionRefinementProjector with text guidance for the refinement process.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes output tokens
4. Processes through multiple refinement steps:
   - Applies self-attention to current features
   - Applies cross-attention to text features
   - Fuses with previous refinement outputs
5. Projects the final output tokens to the target dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection
Output Tokens → [Text-Aware Refinement Block]... → Output Projection
```

**Inspired by**: Progressive refinement with multimodal guidance.

**References**:
- Zhu, X., et al. (2020). Deformable DETR: Deformable Transformers for End-to-End Object Detection. ICLR 2021.
- Alayrac, J.B., et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. NeurIPS 2022.

---

### TextAwareGatedCrossAttentionProjector

**Architecture**: Extends GatedCrossAttentionProjector to use text for conditioning the gating mechanism.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes output tokens
4. Multi-layer processing where each layer:
   - Applies self-attention to output tokens
   - Applies cross-attention from output tokens to visual features with text-conditional gating
   - Applies cross-attention from output tokens to text features
5. Projects the final output tokens to the target dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection → Gate Bias Generator
Output Tokens → [Self-Attention → Text-Gated Cross-Attention with Visual → Cross-Attention with Text]... → Output Projection
```

**Inspired by**: Gated attention mechanisms with language conditioning.

**References**:
- Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL 2019.
- Liu, Y., et al. (2022). Gated Attention as Model Integration for Vision and Language Understanding. ArXiv.

---

### TextAwareMultiScaleContrastiveProjector

**Architecture**: Enhances MultiScaleContrastiveProjector by incorporating text to guide multi-scale processing.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Processes features through multiple scale-specific projections with text conditioning
4. Applies text-aware attention pooling at each scale
5. Computes contrastive losses between scales and with text during training
6. Combines multi-scale representations for the final output

```
Visual Features → Input Projection
Text Tokens → BERT Embeddings → Text Projection
[Scale-Specific Text-Conditioned Projections] → Text-Aware Attention Pooling → Output Projection
```

**Inspired by**: Multimodal contrastive learning techniques.

**References**:
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML 2022.

---

### TextAwarePEVLAdapter

**Architecture**: Extends PEVLAdapter to condition the bottleneck adapters on instruction text.

**Functionality**:
1. Projects input visual features
2. Processes text tokens using BERT embeddings
3. Initializes output tokens
4. Applies text-conditioned bottleneck adapters:
   - Down-projects to a bottleneck dimension based on text features
   - Up-projects back to the original dimension
   - Uses residual connections
5. Applies final attention to aggregate features
6. Projects to the output dimension

```
Visual Features → Visual Projection
Text Tokens → BERT Embeddings → Text Projection
Output Tokens → [Text-Conditioned Adapter Layers]... → Output Attention → Output Projection
```

**Inspired by**: Parameter-efficient adaptation with language conditioning.

**References**:
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. ICML 2019.
- Hu, E.J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Chen, K., et al. (2022). AdapterFormer: Adapting Vision Transformers for Scalable Visual Recognition. NeurIPS 2022.

---

## Comparison of Projector Types

### Complexity and Parameter Count

From lowest to highest complexity:
1. Linear/MLP Projectors - Fewest parameters, simple architecture
2. Self-Aggregating Projectors - Low-medium complexity with attention pooling
3. Cross-Attention Based Projectors - Medium complexity
4. QFormer-based Projectors - Medium-high complexity
5. Perceiver and Refinement Projectors - High complexity
6. MoE and Multi-Scale Projectors - Highest complexity

### Application Scenarios

- **Simple Feature Projection**: LinearProjector, MLPProjector
- **Dense Feature Aggregation**: SelfAggregatingProjectors, CrossAttentionProjector
- **High-Quality Visual Understanding**: QFormerProjector, BLIP2QFormer
- **Handling Long Visual Sequences**: PerceiverProjector, AdaptiveQueryProjector
- **Complex Multi-Step Processing**: FusionRefinementProjector, TextAwareFusionRefinementProjector
- **Compute-Efficient Processing**: HierarchicalMoEProjector, TextAwareHierarchicalMoEProjector
- **Parameter-Efficient Fine-tuning**: PEVLAdapter, TextAwarePEVLAdapter
- **Multi-Resolution Processing**: MultiScaleContrastiveProjector, TextAwareMultiScaleContrastiveProjector

### Text-Aware vs. Base Projectors

Text-aware projectors generally offer these advantages over their base counterparts:
1. **Instructability**: Can process and follow text instructions during inference
2. **Context-Awareness**: Can adapt processing based on provided text context
3. **Finer Control**: Enable more precise control over the visual features being extracted
4. **Better Alignment**: Typically achieve better alignment with the language model

However, they come with these trade-offs:
1. **Increased Complexity**: More parameters and computation required
2. **Dependency on Text**: Rely on high-quality text instructions for optimal performance
3. **More Training Data**: May require more multimodal training data 