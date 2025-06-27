# IMDB 50K Movie Reviews Sentiment Classification

A comprehensive sentiment analysis project that classifies IMDB movie reviews as positive or negative using machine learning techniques. This project compares the performance of XGBoost against traditional linear classifiers on high-dimensional TF-IDF features.

## 🎯 Project Overview

This project implements binary sentiment classification on the IMDB 50K Movie Reviews dataset, focusing on:
- Exploratory Data Analysis (EDA) of review patterns
- Text preprocessing and feature extraction using TF-IDF
- Model comparison between XGBoost, Logistic Regression, and Linear SVC
- Performance evaluation and feature importance analysis

## 📊 Dataset

- **Source:** IMDB 50K Movie Reviews (Kaggle)
- **Size:** 50,000 reviews (25K positive, 25K negative)
- **Features:** Raw text reviews with binary sentiment labels
- **Balance:** Perfectly balanced dataset

## 🔧 Methodology

### 1. Data Preprocessing
- HTML tag removal
- Punctuation and noise cleaning
- Stopword removal
- Lemmatization
- Text normalization

### 2. Feature Engineering
- TF-IDF vectorization with unigrams and bigrams
- Maximum features: 100,000
- Sparse matrix representation

### 3. Model Implementation
- **XGBoost:** Primary gradient boosting classifier
- **Logistic Regression:** Linear baseline model
- **Linear SVC:** Support vector classifier baseline

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Linear SVC** | **90.87%** | **90.50%** | **91.49%** | **90.99%** | **96.88%** |
| **Logistic Regression** | **89.97%** | **88.96%** | **91.45%** | **90.18%** | **96.54%** |
| **XGBoost** | 86.18% | 85.16% | 87.89% | 86.50% | 93.96% |

## 🔍 Key Findings

- **Linear models outperform XGBoost** on high-dimensional TF-IDF features
- Linear SVC achieved the highest accuracy at 90.87%
- All models demonstrate excellent discriminative ability (AUC-ROC ≥ 0.94)
- XGBoost shows strong recall but lower precision compared to linear models
- Linear decision boundaries prove more effective for this text classification task

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **NLP:** NLTK
- **Visualization:** Matplotlib, Seaborn
- **Text Processing:** TF-IDF Vectorization

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **NLP:** NLTK
- **Visualization:** Matplotlib, Seaborn
- **Text Processing:** TF-IDF Vectorization
