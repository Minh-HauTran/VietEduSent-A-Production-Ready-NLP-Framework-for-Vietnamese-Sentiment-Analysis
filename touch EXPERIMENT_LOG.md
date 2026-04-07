# 🧪 Experiment Log – VietEduSent  
*A Research-Oriented Development Record for Vietnamese Sentiment Analysis*

---

## 🎯 Objective

The goal of this project is to design a **production-ready NLP framework** for Vietnamese sentiment analysis, addressing:

- Morphological complexity of Vietnamese (multi-syllable words)
- Severe class imbalance in real-world educational data
- Deployment gap between research models and real-world systems

Dataset: UIT-VSFC (16,175 samples)  
- Positive: 49.7%  
- Negative: 46.0%  
- Neutral: 4.3% (extreme minority) :contentReference[oaicite:1]{index=1}  

---

## 🧱 Phase 1 — Baseline Exploration (Classical ML)

### Models Tested
- Logistic Regression (TF-IDF)
- SVM (Linear)
- Random Forest
- Stacking (LR + SVM)

### Results
| Model | Accuracy | Observation |
|------|--------|------------|
| Logistic Regression | ~90% | Strong linear baseline |
| SVM | ~92–93% | Best classical model |
| Random Forest | Overfit | 99% train / 93% val |
| Stacking | Slight gain | Marginal improvement |

### Key Insight
- Classical models hit a **lexical ceiling (~93%)**
- Unable to capture:
  - word order
  - contextual polarity
  - syntactic dependencies

👉 Conclusion:
> Need contextual embeddings → move to Deep Learning

---

## 🧠 Phase 2 — Deep Learning (Sequential Models)

### Models Tested
- GRU
- BiLSTM

### Results
| Model | Accuracy | Observation |
|------|--------|------------|
| GRU | < 86% | Data insufficient |
| BiLSTM | < 86% | Better context but unstable |

### Key Insight
- RNNs require large datasets → **data starvation problem**
- Vietnamese syntax requires:
  - bidirectional context
  - long-range dependency modeling

👉 Conclusion:
> Pure DL is insufficient → need Transfer Learning

---

## 🤖 Phase 3 — Transformer Models

### Model
- PhoBERT (Vietnamese pretrained model)

### Result
- Accuracy: **96.7%**
- Strong semantic understanding

### Key Insight
- PhoBERT outperforms multilingual models because:
  - word-level tokenization (not syllable-level)
  - preserves semantic integrity

👉 Example:
- "sinh viên" → `sinh_viên` (correct semantic unit)

---

## ⚠️ Critical Problem — Class Imbalance

Neutral class = **4.3% only** :contentReference[oaicite:2]{index=2}  

### Problem
- Cross-Entropy loss:
  - biases toward majority classes
  - ignores neutral class

👉 Model can reach ~95% accuracy by **ignoring Neutral completely**

---

## 🔬 Phase 4 — Optimization Strategy

### Solution: Class-Balanced Focal Loss

Instead of:
- Standard Cross-Entropy

We use:
- Focal Loss + Class weighting

### Effect
- Down-weight easy samples
- Focus on hard / minority cases

### Insight
> Optimization function matters more than architecture in imbalanced datasets

---

## 🚀 Phase 5 — Proposed Hybrid Architecture

### Architecture
**PhoBERT + BiGRU + Attention**

Components:
1. PhoBERT → contextual embeddings  
2. BiGRU → sequential flow  
3. Attention → focus on sentiment-relevant tokens  

### Motivation
- Transformer = global context  
- RNN = sequential bias  
- Attention = selective reasoning  

---

## 📊 Final Results

| Model | Accuracy | Macro F1 |
|------|--------|----------|
| Classical ML | ~93% | Low |
| RNN | <86% | Poor |
| PhoBERT | 96.7% | High |
| Hybrid Model | **97.45%** | **97.38%** |

👉 Improvement is statistically significant (p < 0.01) :contentReference[oaicite:3]{index=3}  

---

## 🧪 Ablation Study

| Component Removed | Accuracy |
|------------------|--------|
| No Focal Loss | ↓ performance |
| No BiGRU | ↓ sequential understanding |
| No Attention | ↓ interpretability |
| PhoBERT only | 96.7% |

### Insight
- Each component contributes meaningfully
- Hybrid architecture is **not redundant**

---

## ⚙️ Training Strategy

- Optimizer: AdamW
- Learning Rates:
  - PhoBERT: 2e-5
  - Head layers: 1e-3
- Validation: Stratified 5-Fold CV
- Early stopping: ~4 epochs

### Insight
> Differential learning rates stabilize fine-tuning

---

## 🔧 Data Processing Pipeline

Steps:
1. Lowercasing  
2. Emoji normalization  
3. Teencode normalization  
4. Regex cleaning  
5. Word segmentation (underthesea)

### Key Insight
Vietnamese requires **word segmentation BEFORE modeling**

Example:
- Raw: "giảng viên"
- Processed: `giảng_viên`

👉 Prevents semantic fragmentation

---

## 🌍 Deployment (Bridging Research → Real World)

- Framework deployed via:
  - HuggingFace Spaces
  - Gradio interface

### Insight
> Most research fails at deployment — this system is production-ready

---

## 🧠 Final Insights

1. **Language matters**  
   → Vietnamese requires specialized NLP pipeline  

2. **Accuracy is misleading**  
   → Must use Macro F1 in imbalanced datasets  

3. **Loss function > model complexity**  
   → Focal Loss solved key bottleneck  

4. **Hybrid architectures win**  
   → Combine Transformer + Sequential + Attention  

5. **Engineering matters**  
   → Deployment is part of the research  

---

## 🔥 Key Takeaway

VietEduSent is not just a model —  
it is a **full-stack NLP system**:

- Data pipeline  
- Model architecture  
- Optimization strategy  
- Evaluation rigor  
- Deployment system  

---
