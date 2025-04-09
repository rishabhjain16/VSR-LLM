# Speech Recognition Evaluation Metrics Guide

This document explains the various metrics used to evaluate speech recognition performance in the VSR-LLM project, with examples showing how each metric behaves for different types of transcription errors.

## Traditional Metrics

### Word Error Rate (WER)
- **Definition**: The percentage of words that are incorrectly recognized, calculated as (Substitutions + Deletions + Insertions) / Total Reference Words × 100%
- **Range**: 0-100%, lower is better (0% = perfect match)
- **Strengths**: Industry standard, easy to understand, sensitive to all word-level errors
- **Weaknesses**: Treats all word errors equally, doesn't account for semantic similarity

**Example 1**: Minor word changes
```
REF: you're hearing one presentation after another often representing a group of people a tribe about how
LLM: you're hearing one person talk after another often representing a group of people in a tribe about how
WER: 18.8%
```
*Analysis*: WER shows word-level differences ("presentation" → "person talk", adding "in") but doesn't recognize semantic similarity.

**Example 2**: Number transcription
```
REF: on september 11 2001 the world changed forever
LLM: on september eleven two thousand and one the world changed forever 
WER: 50.0%
```
*Analysis*: WER is high even though the meaning is identical, because numbers are written differently.

### Character Error Rate (CER)
- **Definition**: The percentage of characters that are incorrectly recognized
- **Range**: 0-100%, lower is better
- **Strengths**: More fine-grained than WER, less affected by tokenization issues
- **Weaknesses**: Doesn't account for word boundaries or semantic similarity

**Example**:
```
REF: last week alone i got 238 pieces of nasty email and more
LLM: last week i got 283 emails and more
WER: 50.0%  CER: 44.4%
```
*Analysis*: CER is slightly lower than WER because character-level differences are proportionally smaller than word-level differences.

## Semantic Metrics

### BERTScore
- **Definition**: Neural metric that uses contextual embeddings to compute similarity between texts
- **Range**: 0-1, higher is better
- **Components**:
  - **Precision (P)**: How much of the hypothesis matches the reference
  - **Recall (R)**: How much of the reference is covered by the hypothesis
  - **F1**: Harmonic mean of precision and recall
- **Strengths**: Captures semantic similarity even when words differ, contextually aware
- **Weaknesses**: Computationally expensive, may not align with human judgments for domain-specific content

**Example**:
```
REF: it turned out that we were doing a lot of low level drug cases on the streets just around the corner from our office in trenton
LLM: it turned out that we were doing a lot of local drug testing on the streets just around the corner from our office in downtown
BERTScore: P=0.8594, R=0.7714, F1=0.8153
```
*Analysis*: Despite word differences ("low level"→"local", "cases"→"testing", "trenton"→"downtown"), BERTScore shows high semantic similarity.

### METEOR
- **Definition**: Metric for Evaluation of Translation with Explicit ORdering that considers exact matches, stemming, synonymy, and paraphrases
- **Range**: 0-1, higher is better
- **Strengths**: Handles synonyms and paraphrases well, correlates better with human judgments than BLEU
- **Weaknesses**: More complex than WER/BLEU, less common in speech recognition

**Example**:
```
REF: you're hearing one presentation after another often representing a group of people a tribe about how
LLM: you're hearing one person talk after another often representing a group of people in a tribe about how
METEOR: 0.9222
```
*Analysis*: METEOR score is high despite word differences because it recognizes "presentation" and "person talk" as semantically related.

### WordSim (Word Similarity)
- **Definition**: Jaccard similarity measuring the overlap between word sets (intersection / union)
- **Range**: 0-1, higher is better
- **Strengths**: Simple, intuitive measure of lexical overlap
- **Weaknesses**: Doesn't consider word order or semantics beyond exact matches

**Example**:
```
REF: last week alone i got 238 pieces of nasty email and more
LLM: last week i got 283 emails and more
WordSim: 0.4286 (word overlap)
```
*Analysis*: Less than half the unique words overlap between reference and hypothesis.

### Semantic WER (SemWER)
- **Definition**: WER-like metric that uses sentence embeddings to calculate semantic distance
- **Components**:
  - **SemWER**: Percentage (lower is better) representing semantic error rate
  - **Similarity**: Raw cosine similarity between sentence embeddings (higher is better)
- **Strengths**: Balances traditional WER approach with semantic understanding
- **Weaknesses**: Depends on quality of underlying embedding model

**Example**:
```
REF: last week alone i got 238 pieces of nasty email and more
LLM: last week i got 283 emails and more
WER: 50.0%  SemWER: 17.2% (sim=0.8275)
```
*Analysis*: SemWER is much lower than WER, correctly identifying that the meaning is largely preserved despite different wording.

### BLEU Score
- **Definition**: BiLingual Evaluation Understudy, measures precision of n-gram matches
- **Range**: 0-100, higher is better
- **Strengths**: Captures phrase-level similarity, standard in machine translation
- **Weaknesses**: Focused on precision, doesn't handle paraphrasing well

**Example**:
```
REF: it turned out that we were doing a lot of low level drug cases on the streets just around the corner from our office in trenton
LLM: it turned out that we were doing a lot of local drug testing on the streets just around the corner from our office in downtown
BLEU: 73.96
```
*Analysis*: High BLEU score indicates good n-gram overlap despite some word substitutions.

### ROUGE Score
- **Definition**: Recall-Oriented Understudy for Gisting Evaluation, measures overlap between generated and reference texts
- **Range**: 0-1, higher is better
- **Variants**:
  - **ROUGE-1**: Overlap of unigrams (single words)
  - **ROUGE-2**: Overlap of bigrams (word pairs)
  - **ROUGE-L**: Longest Common Subsequence, accounting for sentence structure
- **Strengths**: Focuses on recall (measures how much of the reference is captured), considers different levels of text structure
- **Weaknesses**: Doesn't account for synonyms or semantic equivalence unless they share the same words

**Example**:
```
REF: the president said he would veto the bill if it reaches his desk next month
LLM: the president announced that he will reject the legislation if it comes to his desk in the next month
ROUGE-1: 0.6190, ROUGE-2: 0.3333, ROUGE-L: 0.5714
```
*Analysis*: 
- ROUGE-1 (0.6190) shows good single word overlap ("the", "president", "he", "if", etc.)
- ROUGE-2 (0.3333) shows moderate word pair matches are preserved
- ROUGE-L (0.5714) indicates that over half of the reference's structure is maintained

## Comparative Analysis

### Example: Different ways to express numbers
```
REF: in 2022 we saw 9 11 style attacks in multiple countries
LLM: in twenty twenty-two we saw nine eleven style attacks in multiple countries
```

**Metrics comparison**:
- **WER**: ~57% (high error rate due to number format differences)
- **BERTScore**: ~0.92 (high semantic similarity)
- **METEOR**: ~0.85 (recognizes number synonymy)
- **SemWER**: ~8% (recognizes semantic equivalence)
- **ROUGE-L**: ~0.75 (captures structural similarity)
- **BLEU**: ~35 (penalized by n-gram differences)

*Analysis*: Semantic metrics (BERTScore, METEOR, SemWER) correctly identify this as a high-quality transcription despite format differences, while traditional metrics (WER, BLEU) incorrectly indicate poor performance. ROUGE falls in the middle, as it captures more structure than WER but doesn't account for semantic equivalence.

### Example: Missing information
```
REF: the vaccine was tested on 2,500 volunteers across five different countries
LLM: the vaccine was tested on volunteers across countries
```

**Metrics comparison**:
- **WER**: ~36% (misses several words)
- **BERTScore**: P≈0.90, R≈0.70 (high precision but low recall)
- **METEOR**: ~0.65 (penalizes missing information)
- **WordSim**: ~0.60 (only partial word overlap)
- **ROUGE-L**: ~0.60 (detects missing information)
- **SemWER**: ~25% (recognizes partial semantic match)

*Analysis*: The high precision but low recall in BERTScore indicates accurate but incomplete transcription. The model captures the core meaning but misses specific details. ROUGE scores are lower because they emphasize recall, and important details are missing from the hypothesis.

## When to Use Which Metric

- **WER/CER**: Best for exact transcription accuracy evaluation, especially for applications requiring precise wording (medical, legal)
- **BERTScore**: For semantic evaluation where meaning matters more than exact wording
- **METEOR**: Good for handling synonyms and paraphrases
- **WordSim**: Simple indicator of word overlap
- **ROUGE**: For evaluating content coverage, especially useful when recall is important
- **SemWER**: Balances traditional and semantic evaluation
- **BLEU**: Useful for phrase-level evaluation and cross-system comparisons

## Common Patterns in Metrics

1. **WER > SemWER**: Indicates transcription with different wording but preserved meaning
2. **BERTScore P > R**: Model outputs are accurate but incomplete
3. **BERTScore R > P**: Model outputs contain correct information but add extra content
4. **High METEOR + Low BLEU**: Good synonym/paraphrase handling but different phrasing
5. **ROUGE-1 > ROUGE-2**: Good word coverage but different word order/combinations
6. **ROUGE-L > ROUGE-2**: Good overall structure but different local phrasing
7. **High WER + High SemWER**: Genuine semantic error, likely incorrectly transcribed content

## Conclusion

Using these metrics together provides a more comprehensive evaluation of speech recognition quality. Traditional metrics (WER/CER) measure exact transcription accuracy, while semantic metrics capture meaning preservation. Recall-oriented metrics like ROUGE help evaluate content coverage. The best evaluation approach combines all perspectives to fully understand model performance. 