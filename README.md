<div align="center">

# 🔓 ReLoDer — Reconstructing Language from Dense Representations

**Inverting Sentence Embeddings Back Into Human-Readable Text**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Models%20%26%20Datasets-FFD21E)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Can a 1024-dimensional vector remember what it read?*  
*We prove it can.*

---

</div>

## 📌 Overview

**ReLoDer** (Reconstructing Language from Dense Representations) is a research project that explores **embedding inversion** — the task of recovering original natural language text from its dense sentence embedding. While embeddings are widely assumed to be non-invertible "one-way" transformations, this project demonstrates that significant semantic and lexical information can be faithfully reconstructed using a novel architecture based on parallel MLPs and a LoRA-finetuned causal language model.

This work was conducted as part of an internship at **WSAI, IIT Madras**, under the guidance of **Dr. Sudarsun Santhiappan**.

### 🎯 The Core Idea

Given a sentence embedding $e \in \mathbb{R}^{1024}$ produced by an encoder (e.g., `Qwen3-Embedding-0.6B`), can we train a model to generate the original text that produced this embedding?

$$\text{Text} \xrightarrow{\text{Encoder}} e \in \mathbb{R}^{1024} \xrightarrow{\text{ReLoDer}} \hat{\text{Text}}$$

**Answer: Yes.** Our best model achieves a **BERTScore-F1 of 0.88** and a **Cosine Similarity of 0.92** between the original and reconstructed text.

---

## 🏗️ Architecture

The ReLoDer architecture consists of three core components:

```
                    ┌─────────────────────────────────────────────┐
                    │           EndToEndInverter                  │
                    │                                             │
  Sentence          │   ┌───────┐                                 │
  Embedding         │   │ MLP 1 │──→ prefix token 1              │
  (1024-dim)  ──────│──→│ MLP 2 │──→ prefix token 2    ┌───────┐ │
                    │   │ MLP 3 │──→ prefix token 3 ──→│ Qwen3 │──→  Generated
                    │   │  ...  │──→     ...            │ 0.6B  │ │    Text
                    │   │ MLP k │──→ prefix token k    │ +LoRA │ │
                    │   └───────┘                       └───────┘ │
                    └─────────────────────────────────────────────┘
```

### Component Details

| Component | Description |
| :--- | :--- |
| **Encoder** | `Qwen/Qwen3-Embedding-0.6B` — Produces 1024-dim sentence embeddings via mean pooling |
| **k Parallel MLPs** | Each MLP independently maps the 1024-dim embedding to a single decoder-compatible token embedding (`1024 → 2048 → hidden_dim`), producing a sequence of `k` learned "prefix" tokens |
| **Decoder** | `Qwen/Qwen3-0.6B` with LoRA adapters — Takes the `k` prefix embeddings as a soft prompt and autoregressively generates the reconstructed text |

### Training Objective

The model is trained using a **composite loss function**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$

- **$\mathcal{L}_{\text{CE}}$** — Standard causal language modeling cross-entropy loss (teacher-forced on the original text)
- **$\mathcal{L}_{\text{aux}}$** — Auxiliary cosine embedding loss between the mean prefix embedding and the mean text embedding, which acts as a semantic anchor to prevent the prefix tokens from drifting away from the original embedding's meaning
- **$\lambda$** — Empirically optimized to `1.0` via hyperparameter sweep

---

## 📊 Results

### Benchmark Comparison

Evaluated on 5 custom, unseen factual paragraphs across 6 metrics:

| Metric | ReLoDer_v1 (10 Epochs) | ReLoDer_v2 (20 Epochs) |
| :--- | :---: | :---: |
| **ROUGE-1** | 0.4140 | **0.4338** |
| **ROUGE-2** | 0.1070 | **0.1394** |
| **ROUGE-L** | 0.2290 | **0.2520** |
| **BLEU** | 0.0810 | **0.0990** |
| **METEOR** | 0.3144 | **0.3295** |
| **BERTScore-F1** | 0.8815 | **0.8851** |
| **Cosine Similarity** | 0.9520 | **0.9539** |

### Generation Examples

| Original Text | Reconstructed Text |
| :--- | :--- |
| *Paris is the capital of France and it is known for its rich cultural heritage, world-renowned art museums, historic architecture, and famous landmarks like the Eiffel Tower...* | *The city of Paris is known for its rich history, art and culture. It's a world-renowned destination with museums, galleries, theaters, landmarks, architecture, fashion, food... Its iconic landmarks include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral...* |
| *Mount Everest is the highest mountain above sea level, standing at over 8800 meters, and it is located in the Himalayas on the border between Nepal and China...* | *Mount Everest is the highest mountain in the world, located at an altitude of 8,849 meters above sea level. It's a natural wonder... The terrain presents numerous challenges including steep slopes, high altitudes, strong winds, cold temperatures...* |

---

## 📈 Experiment Evolution

The project progressed through a systematic series of architectural and data scaling experiments:

| Version | Prefix Len (k) | LoRA (r, α) | Dataset | Epochs | Key Milestone |
| :--- | :---: | :---: | :---: | :---: | :--- |
| v1–v2 | 8, 16 | 16, 32 | 10K | 10 | Initial proof of concept |
| v3–v6 | 32, 64 | 16, 32 | 10K | 4–10 | Scaling prefix length |
| v7–v8 | 32, 64 | 16, 32 | 100K | 1–2 | Dataset scaling (10x) |
| **ReLoDer_v1** | **64** | **32, 64** | **100K** | **10** | Full end-to-end training on A100 |
| **ReLoDer_v2** | **64** | **32, 64** | **100K** | **20** | Split LR + deeper training |
| **ReLoDer_v3** | **128** | **32, 64** | **1M** | **ongoing** | Massive data + prefix scaling |

### Hyperparameter Sweeps Conducted

**Lambda ($\lambda$) Sweep** — Conducted on 5K records, monitoring CE loss:
| $\lambda$ | CE Loss |
| :---: | :---: |
| 0.25 | 1.8041 |
| 0.50 | 2.3814 |
| **1.00** | **1.7949** ✅ |
| 2.00 | 1.8002 |

**LoRA Rank Sweep** — Conducted on 5K records, monitoring total loss:
| Rank (r) | Total Loss |
| :---: | :---: |
| 4 | 2.0054 |
| **8** | **1.9682** ✅ |
| 16 | 1.9948 |

---

## 📂 Datasets

| Dataset | Records | Source | Link |
| :--- | :---: | :--- | :--- |
| MSMARCO Embeddings (10K) | 10,000 | MS MARCO v1.1 | [Kaggle](https://www.kaggle.com/datasets/subhavkumar/new-msmarco-embeddings-pair-dataset) |
| MSMARCO Qwen Embeddings (100K) | 100,000 | MS MARCO v1.1 | [🤗 HuggingFace](https://huggingface.co/datasets/jg-eno/msmarco-v5.1-Qwen-Embeddings) |
| MSMARCO Qwen Embeddings (1M) | 1,000,000 | MS MARCO v2.1 | [🤗 HuggingFace](https://huggingface.co/datasets/jg-eno/MSMACRO-1M-Qwen-Embeddings) |

Each dataset record contains:
- `sentence_embeddings` — 1024-dim Qwen3-Embedding-0.6B sentence embedding
- `token_embeddings` — Per-token embeddings
- `input_ids` — Tokenized text (Qwen tokenizer)
- `attention_mask` — Attention mask
- `seq_lengths` — Original sequence length
- `texts` — Raw paragraph text

---

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers peft accelerate datasets huggingface_hub evaluate bert_score rouge_score nltk
```

### Inference (Generate Text from Embeddings)

```python
from eval import Runner

configs = [
    {
        "repo": "jg-eno/ReLoDer_v2",
        "prefix_len": 64,
        "filename": "best_checkpoint.pt"
    },
]

texts = [
    "Paris is the capital of France and it is known for its rich cultural heritage.",
    "Mount Everest is the highest mountain above sea level.",
]

runner = Runner()
results = runner.run(configs, texts)
runner.report(texts, results)
```

### Training

```bash
# On a remote GPU server (recommended: A100 or better)
tmux new -s training
python train.py
# Ctrl+B, then D to detach
```

---

## 🔬 Key Technical Decisions

| Decision | Rationale |
| :--- | :--- |
| **k parallel MLPs** instead of a single large MLP | Each MLP independently specializes in generating one prefix token, preventing information bottleneck and improving reconstruction quality |
| **LoRA** instead of full fine-tuning | Reduces trainable parameters by ~97% while retaining the decoder's pretrained language capabilities |
| **Auxiliary cosine loss** | Acts as a semantic anchor that prevents the generated prefix tokens from drifting away from the original embedding's semantic space |
| **Cosine Annealing LR** | Smoothly decays the learning rate to help the optimizer settle into the global minimum during later epochs |
| **Streaming datasets** | The 1M dataset is 40+ GB; streaming from HuggingFace avoids disk space limitations |

---

## 📚 References

This project builds upon insights from the following papers:

1. **Song & Raghunathan (2020)** — *Information Leakage in Embedding Models*
2. **Morris et al. (2023)** — *Text Embeddings Reveal (Almost) As Much As Text*
3. **Li et al. (2023)** — *Sentence Embedding Leaks More Than You Think*
4. **Gu et al. (2024)** — *invBERT: Text Reconstruction from Contextualized Embeddings*
5. **Chen et al. (2024)** — *Zero2Text: Training-Free Zero-Shot Text Reconstruction*
6. **Niu et al. (2025)** — *ZSInvert: Zero-Shot Embedding Inversion*
   
For a detailed analysis, see our [Literature Survey Report](Literature_Survey_Report.md).

---

## 👥 Team

| Name | Role |
| :--- | :--- |
| **Subhav Kumar** | Research Intern, WSAI, IIT Madras |
| **J. Glen Enosh** | Research Intern, WSAI, IIT Madras |
| **Dr. Sudarsun Santhiappan** | Guide, DSAI, IIT Madras |

---

<div align="center">

*Built with ❤️ at WSAI, IIT Madras*

**Project Code:** WSAI/032 — Semantic Text Reconstruction from Embeddings

</div>
