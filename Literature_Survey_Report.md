# Embedding Inversion for Synthetic Data Generation: A Literature Survey




## Abstract

Text embeddings — dense, fixed-dimensional vector representations of natural language — form the backbone of modern NLP systems, powering search engines, retrieval-augmented generation (RAG), and recommendation systems. While embeddings are widely assumed to be non-invertible "one-way" transformations, a growing body of research demonstrates that significant semantic and lexical information can be recovered from these vectors. This survey examines the state of the art in *embedding inversion* — the task of reconstructing the original text from its embedding representation — and evaluates the suitability of these methods for a constructive application: **synthetic data generation**. We review seven core papers spanning 2020–2026, organize them into a three-category taxonomy (Trained Decoder Methods, Zero-Shot Methods, and Diffusion-Based Methods), provide a comprehensive comparative analysis across eight dimensions, and identify the most promising directions for leveraging embedding inversion as a tool for controlled, diverse, and scalable synthetic text generation.

---

## 1. Introduction

### 1.1 Motivation

The rapid adoption of dense text embeddings in production systems has created a vast infrastructure of vector databases storing sensitive information — from personal emails to corporate trade secrets — as high-dimensional vectors. These embeddings are widely considered to be obfuscated representations that preserve utility (enabling semantic search and retrieval) while protecting the underlying text from reconstruction.

However, recent research has systematically challenged this assumption. Beginning with Song & Raghunathan (2020), who demonstrated that individual words can be recovered from embedding vectors, the field has progressed to methods capable of recovering over 90% of the exact original text (Morris et al., 2023). This line of work has primarily been framed as a privacy and security concern.

### 1.2 Reframing: From Privacy Threat to Synthetic Data Tool

This survey adopts an alternative perspective. If embedding inversion methods can reconstruct semantically faithful text from an embedding vector, then the same methods can be repurposed for **synthetic data generation** through manipulation of the embedding space:

| Operation in Embedding Space | Expected Result After Inversion |
|:---|:---|
| Add small Gaussian noise to an embedding | A paraphrase of the original sentence |
| Interpolate between embeddings of A and B | A sentence blending the meanings of A and B |
| Sample from dense regions of embedding space | New sentences in the same topic/style |
| Perform vector arithmetic on embeddings | Semantically controlled text transformation |

This reframing transforms embedding inversion from a purely adversarial technique into a powerful tool for data augmentation, controlled generation, and distribution-aware sampling — applications of significant practical value in NLP.

### 1.3 Scope and Organization

This survey covers seven core papers and two supplementary references, organized as follows:

- **Section 2** presents the technical background required to understand embedding inversion.
- **Section 3** introduces a three-category taxonomy and provides detailed analysis of each method.
- **Section 4** presents a comprehensive comparative analysis across multiple dimensions.
- **Section 5** evaluates each method's suitability for synthetic data generation.
- **Section 6** identifies open challenges and future research directions.

---

## 2. Background and Preliminaries

### 2.1 Sentence Embeddings

Modern sentence embedding models (Reimers & Gurevych, 2019) produce fixed-dimensional dense vectors by encoding variable-length text through a transformer encoder (e.g., BERT) followed by a pooling operation (typically mean pooling over token representations). These models are trained using contrastive objectives such as InfoNCE loss, which enforces that semantically similar sentences produce similar embeddings while dissimilar sentences are pushed apart.

Formally, given an encoder E: X → R^d, a text sequence x is mapped to a d-dimensional embedding vector e = E(x), where d typically ranges from 384 to 4096.

### 2.2 The Embedding Space

The resulting embedding space exhibits several properties critical to both inversion and synthetic data generation:

1. **Approximate Linearity:** Vector arithmetic produces semantically meaningful results (e.g., E("king") - E("man") + E("woman") ≈ E("queen")).
2. **Manifold Structure:** Real text embeddings occupy a lower-dimensional manifold within R^d; not every point in the space corresponds to meaningful text.
3. **Cluster Structure:** Embeddings of related topics form identifiable clusters.
4. **Local Smoothness:** Small perturbations in embedding space correspond to small semantic changes in the decoded text.

### 2.3 Problem Formulation

The embedding inversion problem can be formally stated as:

```
x* = arg min  D(E(x̂), e_target)
      x̂ ∈ X
```

where e_target ∈ R^d is the target embedding vector and D is a distance metric (typically 1 - cosine_similarity). The search space X is combinatorially vast: with vocabulary size V ≈ 30,000 and sequence length T = 32, the space contains V^T ≈ 10^143 possible sequences.

### 2.4 Evaluation Metrics

| Metric | What It Measures |
|:---|:---|
| **Token-level F1** | Precision/Recall of recovered words (bag-of-words overlap) |
| **BLEU-1 / BLEU-2** | N-gram precision between original and reconstructed text |
| **ROUGE-1** | Unigram recall — lexical overlap |
| **ROUGE-L** | Longest Common Subsequence — structural and sequential overlap |
| **Cosine Similarity (COS)** | Semantic similarity between original and reconstructed embeddings |
| **Exact Match** | Percentage of perfectly reconstructed sequences |

---

## 3. Taxonomy of Embedding Inversion Methods

We organize the surveyed methods into three categories based on their architectural paradigm:

```
Embedding Inversion Methods
├── 3.1 Trained Decoder Methods
│     ├── Information Leakage (Song & Raghunathan, 2020)
│     ├── GEIA (Li et al., 2023)
│     └── Vec2Text (Morris et al., 2023)
│
├── 3.2 Zero-Shot / Training-Free Methods
│     ├── ZSInvert (Zhang et al., 2025)
│     ├── ALGEN (Chen et al., 2025)
│     └── Zero2Text (Kim et al., 2026)
│
└── 3.3 Diffusion-Based Methods
      └── Conditional MDLM (Xiao, 2026)
```

---

### 3.1 Trained Decoder Methods

These methods train a dedicated neural network to map from embedding space back to text. They achieve the highest fidelity but require significant training data and are encoder-specific.

#### 3.1.1 Information Leakage in Embedding Models (Song & Raghunathan, 2020)

**Objective:** Recover the bag of words (set of words, ignoring order) from an embedding vector.

**Method:** The authors formulate inversion as an optimization problem. In the white-box setting, the discrete word selection problem is relaxed to a continuous optimization using softmax attention over the vocabulary. A learnable weight vector w ∈ R^V is passed through softmax to produce a "soft" word selection, and gradient descent minimizes the distance between the resulting embedding and the target. For deep models like BERT, a learned linear projection maps embeddings to an intermediate layer representation to improve recovery.

**Key Result:** Achieves 50–70% word recovery in white-box settings. This paper was the first to demonstrate that embeddings are not cryptographically secure.

**Significance:** Established the foundational proof that embedding vectors retain recoverable lexical information, motivating all subsequent work.

#### 3.1.2 GEIA — Generative Embedding Inversion Attack (Li et al., 2023)

**Objective:** Generate the full original sentence (in order) from its embedding.

**Method:** A GPT-2 style autoregressive decoder is trained to generate text conditioned on the input embedding. The embedding is projected through a linear layer and injected as the first token of the decoder's input sequence. The decoder then generates text left-to-right using standard cross-entropy loss.

**Pipeline:** e_target → Linear Projection → pseudo-token → GPT-2 Decoder → x̂₁, x̂₂, ..., x̂_T

**Key Result:** Successfully recovers full sentences with significant semantic overlap, demonstrating that sentence embeddings leak substantially more information than previously expected.

**Limitation:** Requires training a separate decoder for each target encoder model.

#### 3.1.3 Vec2Text (Morris et al., 2023)

**Objective:** Recover the exact original text through iterative correction.

**Method:** Vec2Text introduces a two-model architecture: (1) a T5-based **Hypothesizer** that generates an initial text guess from the target embedding, and (2) a T5-based **Corrector** that takes the concatenation of the current guess's embedding and the target embedding, then generates an improved guess. The difference between the two embedding vectors informs the corrector about what needs to change. The corrector is applied for 50 iterations, progressively closing the gap.

**Key Results:**

| Metric | Score |
|:---|:---|
| Exact Match (32 tokens) | 92% |
| Training Data Required | 5 million (text, embedding) pairs |
| Training Time | 2 days on 4× A6000 GPUs |

**Limitations:** (1) Requires massive encoder-specific training data; (2) Must retrain entirely for each new encoder; (3) Fragile to Gaussian noise; (4) Poor cross-domain generalization.

**Significance:** Established the performance benchmark that all subsequent methods compare against.

---

### 3.2 Zero-Shot / Training-Free Methods

These methods eliminate or drastically reduce the need for encoder-specific training, trading some accuracy for universality and practicality.

#### 3.2.1 ZSInvert — Universal Zero-Shot Embedding Inversion (Zhang et al., 2025)

**Objective:** Invert embeddings from any encoder without per-encoder training.

**Method:** ZSInvert introduces a three-stage pipeline:

- **Stage 1 — Seed Generation:** An LLM generates text via beam search with the vague prefix "tell me a story." Candidates are scored purely by cosine similarity to the target embedding: score(x̂) = cos(E(x̂), e_target). The top-k tokens from the LLM's distribution are expanded, encoded, and scored; the top-b beams are retained.

- **Stage 2 — Paraphrase Refinement:** The prefix is changed to "write a sentence similar to: \<Stage 1 output\>," and adversarial decoding is repeated, narrowing the search to the correct semantic neighborhood.

- **Stage 3 — Correction Model:** A Qwen2.5-3B model fine-tuned on only 400 examples performs multi-candidate denoising: given 5 imperfect inversions, it predicts the original text. Trained once on Contriever, it transfers universally because it learns a linguistic skill rather than an encoder-specific mapping.

Stages 2–3 repeat for 6–9 iterations with a growing candidate list.

**Key Results:**

| Encoder | F1 Score | Cosine Similarity | Info Leakage (Enron) |
|:---|:---|:---|:---|
| Contriever | 59.54 | 81.41% | 86% |
| GTR | 54.39 | 87.38% | 88% |
| GTE | 52.93 | 94.36% | 82% |
| GTE-Qwen | 50.41 | 80.80% | 92% |

Noise robust up to σ = 0.01 Gaussian noise.

#### 3.2.2 ALGEN — Few-Shot Cross-Model Alignment (Chen et al., 2025)

**Objective:** Invert embeddings using a proxy encoder with minimal leaked data for alignment.

**Method:** A two-phase approach: (1) train a Flan-T5-small decoder on 150K pairs from a free proxy encoder, then (2) train a small adapter on ~1,000 leaked victim pairs to translate victim embeddings into the proxy's space.

**Key Limitation:** The adapter fails to generalize when the target embedding originates from an unseen domain, making it unsuitable for cross-domain settings.

#### 3.2.3 Zero2Text (Kim et al., 2026)

**Objective:** Invert embeddings with zero training, zero leaked data, and robust cross-domain generalization.

**Method:** Zero2Text introduces **recursive online alignment** built on two principles:

1. **LLM as Universal Generator:** Qwen3-0.6B generates diverse candidate tokens at each step, filtered for diversity (pairwise cosine similarity < 0.9 threshold).

2. **Online Projection Optimization:** A projection matrix W_t mapping local encoder embeddings to the victim's space is learned on-the-fly via Ridge Regression:

```
W_t = (E_t^T · E_t + λI)^(-1) · E_t^T · Ẽ_t
```

where E_t contains local embeddings and Ẽ_t contains victim embeddings accumulated through API queries. W_t is recalculated at each step, becoming progressively accurate.

**Scoring Function:** S(e_i, t) = Z(y_i) + conf_t × Z(cos(e_i, e_v)), where conf_t dynamically weights embedding similarity based on the current accuracy of W_t.

**Query Efficiency:** Exponential decay (γ = 0.8) reduces API queries per step, totaling ~2,180 queries and ~14K tokens per inversion.

**Key Results (MS MARCO, cross-domain, OpenAI-3-large victim):**

| Method | BLEU-2 | ROUGE-L | COS |
|:---|:---|:---|:---|
| Vec2Text (w/ Corrector) | 2.44 | 8.91 | 0.10 |
| ALGEN | 2.38 | 14.79 | 0.25 |
| **Zero2Text** | **16.42** | **26.08** | **0.66** |

Achieves 1.8× higher ROUGE-L and 6.4× higher BLEU-2 than the strongest baseline.

---

### 3.3 Diffusion-Based Methods

#### 3.3.1 Conditional Masked Diffusion Language Model (Xiao, 2026)

**Objective:** Invert embeddings using parallel generation with inherent output diversity.

**Method:** A Masked Diffusion Language Model (MDLM) is trained to predict masked tokens conditioned on the target embedding via **Adaptive Layer Normalization (AdaLN)**, where the normalization parameters γ and β are derived from the embedding vector, injecting semantic information at every transformer layer.

**Inference:** Starting from a fully masked sequence, the model iteratively unmasks the most confident tokens over ~8 denoising steps, generating all positions in parallel.

**Key Property — Stochastic Diversity:** Unlike autoregressive methods, diffusion-based generation is inherently stochastic — running the same inversion multiple times produces different but semantically consistent outputs. This is the ideal property for synthetic data generation.

**Key Results:** 81.3% token-level accuracy, encoder-agnostic transfer, fast inference (8 forward passes).

---

## 4. Comparative Analysis

### 4.1 Training and Data Requirements

| Method | Training | Data Required | Time | Per-Encoder? |
|:---|:---|:---|:---|:---|
| Info Leakage | Optimization only | None | N/A | Yes (white-box) |
| GEIA | Decoder training | Thousands of pairs | Hours | Yes |
| Vec2Text | Heavy (2 models) | 5M pairs | 2 days, 4× GPU | Yes |
| ZSInvert | Minimal (correction) | 400 pairs | 10 min, 1× GPU | No |
| ALGEN | Decoder + Adapter | 150K + 1K leaked | Hours | Partially |
| Zero2Text | **None** | **None** | **Zero** | **No** |
| Diffusion | Model training | Large corpus | Hours | No |

### 4.2 Performance Comparison

| Method | Best Accuracy | Cross-Domain | Noise Robust | Black-Box |
|:---|:---|:---|:---|:---|
| Info Leakage | 50–70% word recovery | N/A | N/A | No |
| GEIA | Moderate recovery | No | No | No |
| Vec2Text | **92% exact match** | No | No | Partial |
| ZSInvert | F1 ≈ 55 | Partial | Yes | Yes |
| ALGEN | ROUGE-L ≈ 15 | No | Limited | Partial |
| Zero2Text | ROUGE-L ≈ 26 | **Yes** | Yes | **Yes** |
| Diffusion | 81.3% token accuracy | Yes | TBD | Yes |

### 4.3 Suitability for Synthetic Data Generation

| Method | Diverse Outputs? | Handles Manipulated Embeddings? | Speed | Verdict |
|:---|:---|:---|:---|:---|
| Info Leakage | No | No | Slow | ❌ Not suitable |
| GEIA | Limited | Somewhat | Fast | ⚠️ Limited |
| Vec2Text | No (deterministic) | Somewhat | Slow | ⚠️ Limited |
| ZSInvert | Limited | Yes (noise robust) | 90 sec | ✅ Good |
| ALGEN | No | No (domain-locked) | Fast | ❌ Not suitable |
| Zero2Text | Limited | Yes (any domain) | ~2K queries | ✅ Good |
| **Diffusion** | **Yes (stochastic)** | **Yes** | **Fast (8 steps)** | **⭐ Best** |

---

## 5. Application to Synthetic Data Generation

### 5.1 How Embedding Inversion Enables Synthetic Data

The embedding space's properties make it a natural medium for controlled text generation:

- **Interpolation:** The interpolated embedding e_α = (1-α)·e_A + α·e_B can be inverted to produce text blending the semantics of both sources.
- **Perturbation:** Adding controlled noise e' = e + N(0, σ²I) produces paraphrases with controllable diversity.
- **Arithmetic:** Vector operations followed by inversion yield semantically transformed text.
- **Sparse Region Sampling:** Generating embeddings in underrepresented areas produces data for dataset balancing.

### 5.2 Proposed Synthetic Data Pipeline

```
1. Embed existing dataset using a sentence encoder
2. Identify target regions for augmentation (sparse areas, class boundaries)
3. Generate synthetic embeddings via interpolation, perturbation, or arithmetic
4. Invert synthetic embeddings using a suitable method (Diffusion or Zero2Text)
5. Filter and validate generated text (BERTScore, MAUVE, human evaluation)
6. Augment training data with validated synthetic samples
```

---

## 6. Open Challenges and Future Directions

1. **Scaling Beyond 32 Tokens:** Nearly all methods are evaluated on short texts (≤32 tokens). Extending to paragraph- and document-level text remains an open problem.

2. **Diversity-Fidelity Tradeoff:** Higher diversity typically reduces semantic fidelity. Developing principled methods to control this tradeoff is critical.

3. **Quality Evaluation:** Synthetic data quality should be measured by downstream task performance and distributional coverage (MAUVE), not just proximity to originals.

4. **On-Manifold Generation:** Manipulated embeddings may lie off the natural data manifold, leading to incoherent inversions. Manifold projection before inversion could improve quality.

5. **Computational Efficiency:** Current methods require 30 seconds to several minutes per sample. Batched inference and further optimization are needed for large-scale generation.

---

## 7. Conclusion

This survey has examined the rapidly evolving field of embedding inversion through the lens of synthetic data generation. The field has progressed from white-box bag-of-words recovery (2020) to training-free, cross-domain, black-box sentence reconstruction (2026). Our analysis identifies Diffusion-based approaches as the most promising paradigm for synthetic data generation due to their inherent output diversity and parallel generation, while Zero2Text and ZSInvert offer practical training-free alternatives for immediate deployment across diverse encoder architectures.

---

## References

1. Song, C. & Raghunathan, A. (2020). "Information Leakage in Embedding Models." *ACM CCS*, pp. 377–390.
2. Li, H., Xu, M. & Song, Y. (2023). "Sentence Embedding Leaks More Information Than You Expect." *Findings of ACL 2023*.
3. Morris, J. et al. (2023). "Text Embeddings Reveal (Almost) As Much As Text." *EMNLP 2023*, pp. 12448–12460.
4. Zhang, C., Morris, J.X. & Shmatikov, V. (2025). "Universal Zero-Shot Embedding Inversion." *arXiv:2504.00147*.
5. Chen, Y., Xu, Q. & Bjerva, J. (2025). "ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment and Generation." *ACL 2025*.
6. Kim, D. et al. (2026). "Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings." *arXiv:2602.01757*.
7. Xiao, H. (2026). "Conditional Masked Diffusion Language Models for Embedding Inversion." *Jina AI Technical Report*.
8. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*.
9. Kugler, K. et al. (2022). "InvBERT: Reconstructing Text from Contextualized Word Embeddings." *arXiv:2109.10104*.
