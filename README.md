# AI vs Human Text Detection - Deep Learning Benchmark

Comparing 5 deep learning architectures for binary text classification on the task of detecting AI-generated vs. human-written text.

---

## Project Overview

With the rise of large language models like ChatGPT, detecting AI-generated text has become a critical challenge in education, journalism, and content moderation. This project builds and compares five distinct deep learning architectures trained from scratch on a 10,000-sample dataset to classify text as either human-written or AI-generated.

---

## Results Summary

| Model | Accuracy | F1 (Human) | F1 (AI) | Training Time |
|---|---|---|---|---|
| Custom Transformer Encoder | 0.984 | 0.99 | 0.98 | 394s |
| 1D-CNN | 0.981 | 0.98 | 0.97 | 332s |
| Gated Convolution Unit (GCU) | 0.9775 | 0.98 | 0.97 | 133s |
| Hybrid (LSTM + Numeric) | 0.9725 | 0.98 | 0.96 | 603s |
| Bidirectional LSTM | 0.957 | 0.97 | 0.94 | 1178s |

The Transformer achieves 98.4% accuracy with the fewest misclassifications overall.

---

## Architectures Implemented

Five architecturally distinct models were compared under identical training conditions:

| Architecture | Key Idea |
|---|---|
| Bidirectional LSTM | Sequential processing in both directions |
| 1D-CNN | Parallel convolutional filters detecting local n-gram patterns |
| Gated Convolution Unit (GCU) | CNN with sigmoid gating to filter relevant features |
| Hybrid (LSTM + Numeric) | Multimodal: text sequences + 5 engineered numeric features |
| Custom Transformer Encoder | Self-attention for capturing long-range dependencies |

DistilBERT (~0.99 accuracy) was tested as an optional experiment but excluded from the fair comparison since it uses pretrained weights, unlike all five models above which are trained from scratch.

---

## Dataset

- Source: ai_human_10k.csv, 10,000 labeled text samples
- Labels: 0 = Human-written, 1 = AI-generated
- Distribution: 6,276 Human / 3,724 AI (mild imbalance)
- Split: 80% Train / 20% Test (stratified)
- Dataset available at: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

---

## Preprocessing Pipeline

1. Text Cleaning: remove markdown, HTML tags, quotes, non-alphanumeric noise, normalize whitespace, lowercase
2. Tokenization: vocabulary size 20,000, fitted on training data only (no data leakage)
3. Padding: all sequences padded to 500 tokens
4. Numeric Feature Engineering (for Hybrid model):
   - Character count
   - Word count
   - Average word length
   - Sentence count
   - Punctuation ratio
5. Scaling: StandardScaler fitted on training split only

---

## Fair Training Setup

All models trained under identical conditions to ensure architecture is the only variable:

| Setting | Value |
|---|---|
| Embedding Dimension | 64 |
| Sequence Length | 500 |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 32 |
| Max Epochs | 5 |
| Early Stopping | patience=2 on val_loss |
| Tokenizer | Same for all models |
| Train/Test Split | Same random_state=42 |

---

## Project Structure

```
ai-human-text-detection/
|
|-- ai_vs_human_text_detection.ipynb   # Full notebook: EDA, Models, Results
|-- ai_human_10k.csv                   # Dataset (not included, see link above)
|-- README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Uththara15/ai-human-text-detection.git
cd ai-human-text-detection

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

# Add the dataset
# Place ai_human_10k.csv in the root directory

# Run the notebook
jupyter notebook ai_vs_human_text_detection.ipynb
```

---

## Key Findings

- Transformers perform best for long-text classification as self-attention captures both local and global context
- CNNs are fast to train and generalize well through local n-gram feature detection
- GCU improves on CNN with gating that filters out irrelevant features
- Hybrid models benefit from linguistic features but require careful tuning to avoid overfitting
- LSTMs struggle with long sequences due to vanishing gradients over long-range dependencies
- Class weights improve Transformer performance most but hurt LSTM accuracy

---

## Tech Stack

- Python 3.10
- TensorFlow 2.x / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## Author

Madee Uththara Deegoda Gamage
Bachelor of Engineering in ICT (Data Analytics & AI)
JAMK University of Applied Sciences, Finland

GitHub: https://github.com/Uththara15
LinkedIn: https://linkedin.com/in/uththara15
