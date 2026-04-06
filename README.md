# 🧠 VietEduSent: A Production-Ready NLP Framework for Vietnamese Sentiment Analysis

## 📌 Abstract
This project presents a production-ready Natural Language Processing (NLP) framework for Vietnamese sentiment analysis. The system integrates traditional machine learning models and state-of-the-art transformer architectures (PhoBERT) to analyze student feedback effectively. Experimental results demonstrate that PhoBERT achieves superior performance, reaching **96.7% accuracy** on the UIT-VSFC dataset.

---

## 🏗 System Architecture
Raw Data → Preprocessing → Tokenization → Feature Engineering → Models → Evaluation → Deployment

---

## 📊 Dataset

- **Dataset:** UIT-VSFC (Vietnamese Students Feedback Corpus)
- Source: HuggingFace
- Samples: ~16,000
- Labels:
  - 0: Negative
  - 1: Neutral
  - 2: Positive

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Lowercasing text
- Removing punctuation & emojis
- Normalization using `underthesea`
- Vietnamese word tokenization
- Cleaning noisy patterns

Example:
"giảng viên rất nhiệt tình!!!"
→ "giảng_viên rất nhiệt_tình"


---

## 🤖 Models

### Traditional ML
- Logistic Regression
- SVM
- Random Forest
- Ensemble Learning

### Deep Learning
- GRU
- BiLSTM

### Transformer
- **PhoBERT (Best Model)**

---

## 📈 Results

| Model | Accuracy |
|------|--------|
| PhoBERT | **96.70%** |
| Random Forest | 93.67% |
| GRU | 93.35% |
| BiLSTM | 92.14% |
| SVM | 90.58% |

---

## 🏆 Best Model: PhoBERT

PhoBERT achieves state-of-the-art performance due to its strong contextual understanding for Vietnamese language.

---

## 🚀 Deployment

- Gradio Web App
- HuggingFace Spaces

Features:
- Single prediction
- Batch prediction (CSV/Excel)
- Visualization (Pie chart)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/vietedusent
cd vietedusent

pip install -r requirements.txt

## ▶️ Run
python app/app.py
