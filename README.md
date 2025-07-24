# Credit Card Fraud Detection

This project focuses on building machine learning models to detect fraudulent transactions in a large, real-world financial dataset. The goal is to distinguish between legitimate and fraudulent transfers using supervised learning techniques.

The dataset is heavily imbalanced, with fraud cases making up a very small percentage of the total records, which presents unique modeling and evaluation challenges.

## Dataset

The dataset used contains **6,362,620 rows** and **11 columns**, with each row representing a financial transaction.

Key features include:

- `type`: Type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT)
- `amount`: Transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: Origin account balances before and after the transaction
- `oldbalanceDest`, `newbalanceDest`: Destination account balances before and after the transaction
- `isFraud`: Target variable (1 if fraud, 0 otherwise)

The dataset is publicly available and widely used in fraud detection research.

## Exploratory Data Analysis

A few key visualizations were used to better understand the data:

- **Transaction Type Distribution**: Most transactions are of type `CASH_OUT` and `TRANSFER`, both in count and total amount.
- **Fraud Ratio**: Only a tiny fraction of transactions are fraudulent (~0.1291%), which makes the dataset highly imbalanced.
- **Correlation Heatmap**: A correlation matrix was generated to examine relationships between features after encoding categorical values.

These insights guided decisions in feature selection and model design.

## Data Preprocessing & Feature Engineering

Key steps:

- **One-Hot Encoding**: The `type` column was encoded using one-hot encoding to convert categorical values into numeric features.
- **Dropping Non-useful Columns**: Columns like `nameOrig`, `nameDest`, and `isFlaggedFraud` were dropped due to irrelevance or high cardinality.
- **Feature Scaling**: Numeric features such as `amount`, `oldbalanceOrg`, and `newbalanceDest` were standardized using `StandardScaler`.
- **Train-Test Split**: The data was split with 70% for training and 30% for testing.

## Model Training

Three classification models were trained and evaluated:

- **Logistic Regression**
- **XGBoost Classifier**
- **Random Forest Classifier**

Each model was trained on the preprocessed data and evaluated on both training and testing sets. The primary metric used was **accuracy**, along with additional classification reports for the XGBoost model.

## Model Evaluation

The best performing model was the **XGBoost Classifier**, which achieved:

- **Accuracy**: ~99.9%
- **F1-score (Fraud Class)**: 0.90
- **Precision**: 0.96
- **Recall**: 0.86

A confusion matrix and full classification report were generated to analyze performance in detail, especially on the minority fraud class.

Despite the class imbalance, the model successfully learned to detect fraudulent transactions with high precision and recall.
