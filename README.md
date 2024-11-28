# Fraud Detection using Machine Learning and Genetic Algorithm

## Overview
This project demonstrates a fraud detection engine for credit card transactions using machine learning classifiers and Genetic Algorithm (GA) for feature selection. The dataset is sourced from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains anonymized features resulting from PCA transformation. The response variable `Class` indicates fraud (1) or non-fraud (0).

---

## Dataset Details
- **Transactions**: 284,807
- **Frauds**: 492 (0.172% of all transactions)
- **Features**:
  - `V1`, `V2`, ..., `V28`: Principal components from PCA.
  - `Time`: Seconds elapsed between transactions.
  - `Amount`: Transaction amount.
  - `Class`: Target variable (1 for fraud, 0 for non-fraud).

---

## Project Pipeline
1. **Data Preprocessing**:
   - Normalize inputs using Min-Max Scaling.
   - Handle imbalanced data using the raw dataset for evaluation.

2. **Feature Selection**:
   - Utilize a Genetic Algorithm (GA) to select the most relevant features for fraud detection.

3. **Machine Learning Classifiers**:
   - Decision Tree (DT)
   - Random Forest (RF)
   - Logistic Regression (LR)
   - Artificial Neural Network (ANN)
   - Naive Bayes (NB)

4. **Evaluation Metrics**:
   - **Accuracy**
   - **Recall**
   - **Precision**
   - **F1-Score**
   - **AUC (Area Under the Curve)**

5. **Visualization**:
   - GA convergence curve.
   - Bar graph comparison of evaluation metrics.
   - Line graph for AUC scores.

---

## Requirements
Install the required libraries using:
```bash
pip install -r requirements.txt
