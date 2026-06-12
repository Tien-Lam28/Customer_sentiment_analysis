# Vietnamese Customer Review Sentiment Analysis

## Project Overview

Vietnamese customer reviews contain valuable signals about customer satisfaction, product quality, service issues, and brand perception. However, manually reading and classifying large volumes of reviews is slow, inconsistent, and difficult to scale.

This project builds an end-to-end Natural Language Processing pipeline to classify Vietnamese customer reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**. The repository is organized for reproducible experimentation, model comparison, evaluation, and final reporting.

## Objectives

- Collect and organize Vietnamese customer review data.
- Clean noisy review text and normalize Vietnamese-specific language patterns.
- Apply Vietnamese word segmentation and tokenization.
- Engineer classical and neural text features.
- Train and compare baseline machine learning, deep learning, and transformer models.
- Evaluate models using standard classification metrics.
- Perform error analysis to understand failure cases.
- Produce a final project report with insights and recommendations.

## Dataset Description

| Field | Description |
| ----- | ----------- |
| Source | Placeholder: e-commerce platforms, app reviews, survey responses, or public Vietnamese review datasets |
| Number of Samples | Placeholder: to be updated after data collection |
| Features | Review text, optional metadata such as product category, rating, date, or platform |
| Labels | Positive, Neutral, Negative |

## NLP Pipeline Diagram

```text
Raw Reviews
    ↓
Cleaning
    ↓
Normalization
    ↓
Word Segmentation
    ↓
Tokenization
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Evaluation
    ↓
Error Analysis
```

## Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- LSTM
- PhoBERT

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Repository Structure

```text
sentiment-analysis-vietnamese/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_svm_model.ipynb
│   ├── 07_lstm_model.ipynb
│   ├── 08_phobert_finetuning.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── load_data.py
│   ├── preprocessing/
│   │   ├── clean_text.py
│   │   ├── normalize_text.py
│   │   ├── word_segmentation.py
│   │   └── tokenizer.py
│   ├── features/
│   │   ├── tfidf_features.py
│   │   ├── word2vec_features.py
│   │   └── fasttext_features.py
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_naive_bayes.py
│   │   ├── train_svm.py
│   │   ├── train_lstm.py
│   │   └── train_phobert.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
│
├── models/
│   ├── baseline/
│   ├── svm/
│   ├── lstm/
│   └── phobert/
│
└── references/
    ├── papers/
    └── notes/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks for experimentation:

```bash
jupyter notebook notebooks/
```

Prepare datasets:

```bash
python src/data/make_dataset.py
```

Train baseline models:

```bash
python src/models/train_logistic_regression.py
python src/models/train_naive_bayes.py
python src/models/train_svm.py
```

Evaluate trained models:

```bash
python src/evaluation/metrics.py
python src/evaluation/error_analysis.py
```

## Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression |          |          |
| SVM                 |          |          |
| LSTM                |          |          |
| PhoBERT             |          |          |

## Future Improvements

- Expand dataset coverage across domains and review platforms.
- Add active labeling guidelines for consistent sentiment annotation.
- Compare additional Vietnamese embeddings and transformer checkpoints.
- Add experiment tracking with MLflow or Weights & Biases.
- Add model explainability for production-facing insights.
- Package the final pipeline for inference once deployment becomes in scope.

## References

- Placeholder: Vietnamese NLP papers and benchmark datasets.
- Placeholder: PhoBERT documentation and related publications.
- Placeholder: Vietnamese word segmentation references.

## Author

**Author:** Nhom 1 KDL24

