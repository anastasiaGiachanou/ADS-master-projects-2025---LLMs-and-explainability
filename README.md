# 🧠 Explainable Prompt Engineering with SHAP and Large Language Models

This repository explores how **explainable AI techniques**, particularly SHAP, can guide prompt construction and evaluation in large language models (LLMs). The focus is on improving interpretability and robustness using the GSM8K dataset and analyzing how prompt variants (focused cues, distractors) affect model performance.

---

## 🔍 Project Highlights

- **Dataset**: gsm8k subset of 300 examples.
- **Models Used**:
  - `GPT-2`: SHAP attribution + perplexity-based fluency scoring.
  - `Qwen2.5-7B-Instruct`: Main model for generating reasoning outputs.
  - `SentenceTransformer` (`all-MiniLM-L6-v2`): To compute cosine similarity between gold and model reasoning.
- **Explainability**: SHAP is used to extract the most influential tokens driving GPT-2’s predictions.
- **Evaluation Dimensions**:
  - **Accuracy**: Whether the predicted answer matches the gold answer.
  - **Fluency**: Measured via perplexity using GPT-2.
  - **Reasoning Similarity**: Semantic closeness to the gold chain-of-thought, via cosine similarity.

---

## 💻 Reproducibility & Environment Notes

- 🧪 **Reproducibility**: All sampling (e.g., data selection, distractor injection) is controlled with `seed=42`.
- 💻 **Tested On**:
  - **MacBook Pro M1 (2021)** — 32GB. MPS acceleration is *not* sufficient for large model inference.
  - **Google Colab** with A100 GPU runtime (recommended for Qwen2.5-7B or larger models).
- ⚡ **GPU Usage**: Ensure `torch.cuda.is_available()` is true for best performance.

---

## 🛠️ Installation

Install necessary packages:

```bash
pip install --upgrade datasets sentence-transformers shap transformers
