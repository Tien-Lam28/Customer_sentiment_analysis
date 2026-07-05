# Customer Sentiment Analysis

A Vietnamese customer review sentiment classification system built with classical ML, LSTM, and PhoBERT fine-tuning. The pipeline covers the full NLP workflow — from raw text cleaning to model evaluation — applied to Vietnamese e-commerce review data.

---

## Table of Contents

- [Customer Sentiment Analysis](#customer-sentiment-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Pipeline](#pipeline)
  - [Models](#models)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Prepare data](#1-prepare-data)
    - [2. Run preprocessing](#2-run-preprocessing)
    - [3. Train models](#3-train-models)
    - [4. Evaluate](#4-evaluate)
  - [Results](#results)
  - [Team](#team)

---

## Overview

Vietnamese customer reviews contain rich signals about product quality, service satisfaction, and brand perception. This project builds an end-to-end NLP pipeline to classify reviews into **Positive** and **Negative** sentiment categories.

**Dataset:**

| Field   | Detail |
|---------|--------|
| Domain  | Vietnamese e-commerce product reviews |
| Labels  | Positive, Negative |
| Features | Review text, star rating, media presence |

**Evaluation Metrics:** Accuracy · Precision · Recall · F1 Score · Confusion Matrix

---

## Pipeline

```
Raw Reviews
    ↓
Text Cleaning       — lowercase, remove URLs, HTML, special characters
    ↓
Normalization       — Vietnamese-specific character and word normalization
    ↓
Word Segmentation   — underthesea / pyvi
    ↓
Feature Engineering — TF-IDF / Word2Vec / FastText / PhoBERT Embeddings
    ↓
Model Training      — Logistic Regression, Naive Bayes, SVM, LSTM, PhoBERT
    ↓
Evaluation          — Metrics, confusion matrix, error analysis
```

---

## Models

| Model | Type | Input Features |
|---|---|---|
| Logistic Regression | Classical ML | TF-IDF |
| Naive Bayes | Classical ML | TF-IDF |
| SVM | Classical ML | TF-IDF |
| LSTM | Deep Learning | Word Embeddings |
| PhoBERT | Transformer | Subword Tokens |

---

## Project Structure

```
Customer_sentiment_analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Original review data
│   ├── interim/                # Cleaned text (clean_reviews.csv)
│   ├── processed/              # Extracted features (.pkl files)
│   └── external/               # External resources (stopwords, dictionaries)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_lstm_model.ipynb
│   └── 07_phobert_finetuning.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_text.py       # Lowercase, remove URLs, HTML, punctuation
│   │   ├── normalize_text.py   # Vietnamese text normalization
│   │   ├── word_segmentation.py
│   │   └── tokenizer.py        # TF-IDF and PhoBERT tokenizer wrappers
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
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── models/
│   ├── baseline/               # Saved Logistic Regression, NB, SVM artifacts
│   ├── lstm/
│   └── phobert/
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
└── references/
    ├── papers/
    └── notes/
```

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/your-org/Customer_sentiment_analysis.git
cd Customer_sentiment_analysis
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `transformers`, `underthesea`, `scikit-learn`, `gensim`, `pandas`

---

## Usage

### 1. Prepare data

```bash
python src/data/make_dataset.py
```

### 2. Run preprocessing

```bash
# Or work interactively through the notebook
jupyter notebook notebooks/03_text_preprocessing.ipynb
```

### 3. Train models

```bash
# Classical baselines
python src/models/train_logistic_regression.py
python src/models/train_naive_bayes.py
python src/models/train_svm.py

# Deep learning
python src/models/train_lstm.py
python src/models/train_phobert.py
```

### 4. Evaluate

```bash
python src/evaluation/metrics.py
python src/evaluation/error_analysis.py
```

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | — | — |
| Naive Bayes | — | — |
| SVM | — | — |
| LSTM | — | — |
| PhoBERT | — | — |

> Results will be updated after full training runs.

---

## Team

| Member | Responsibility |
|---|---|
| Diệu Thùy | Data Understanding & EDA (`01`, `02`) |
| Nhật Tiến, Quang Phong | Text Preprocessing (`03`) |
| Hưng Trương | Feature Engineering (`04`) |
| Quốc Triều | Baseline ML Models (`05`) |
| Ngọc Tiến, Hoàng Sang | Deep Learning — LSTM & PhoBERT (`06`, `07`) |
