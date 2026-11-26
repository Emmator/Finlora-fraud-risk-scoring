# Finlora Fraud Risk Scoring System

### Machine Learning-Based Transaction Fraud Detection

Finlora is a supervised machine-learning project designed to identify potentially fraudulent financial transactions and support risk-based transaction review.

The project covers data cleaning, exploratory data analysis, feature engineering, model development, evaluation, error analysis, and deployment of a Streamlit fraud risk scoring prototype.

## 1. Project Objectives

- Develop models to distinguish fraudulent from legitimate transactions.
- Evaluate fraud detection performance under class imbalance.
- Compare Logistic Regression and Random Forest models.
- Analyze model errors, transaction segments, and operational trade-offs.
- Develop an interactive Streamlit application for transaction scoring and analyst review.

## 2. Dataset

The project uses a synthetic financial transaction dataset.

| Characteristic | Value |
|---|---|
| Total transactions | 126,000 |
| Fraudulent transactions | 3,352 |
| Fraud prevalence | 2.66% |
| Customer accounts | 7,200 |
| Problem type | Binary classification |

The dataset is imbalanced, making precision, recall, F1-score, and average precision particularly important evaluation metrics.

## 3. Project Workflow

The machine-learning workflow consists of:

1. Data cleaning and validation.
2. Exploratory data analysis.
3. Feature engineering and preprocessing.
4. Logistic Regression and Random Forest training.
5. Model performance evaluation.
6. Threshold and operational workload analysis.
7. Error analysis and temporal robustness testing.
8. Streamlit application development.

## 4. Model Performance

Initial baseline models were evaluated using a held-out random test dataset of 31,500 transactions.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.2582 | 0.3198 |
| Recall | 0.8640 | 0.8258 |
| F1-score | 0.3976 | 0.4610 |
| ROC-AUC | 0.9394 | 0.9317 |
| Average Precision | 0.7102 | 0.5914 |

Logistic Regression achieved higher recall and average precision, whereas Random Forest achieved higher precision and F1-score at the original baseline decision threshold.

These results represent experimental performance on the synthetic dataset and should not be interpreted as production fraud detection guarantees.

## 5. Feature Engineering

The trained Random Forest receives 18 input features, which are transformed into 42 processed features.

Key inputs include:

- Transaction amount and historical spending ratio.
- Transaction velocity within one hour.
- Transaction hour.
- Cross-border transaction indicator.
- Device status.
- Account type and KYC tier.
- Currency, merchant category, and transaction channel.

Additional engineered features capture logarithmic transaction values and cyclical time patterns.

## 6. Streamlit Application

The project includes a Streamlit-based fraud risk scoring prototype with two main capabilities.

**Single Transaction Scoring**

Users can enter transaction characteristics, calculate engineered features, and generate a Random Forest risk score with a demonstration review recommendation.

**Fraud Review Queue**

Users can upload a cleaned transaction CSV, generate risk scores for multiple transactions, filter review recommendations, and download scoring results.

The application uses a demonstration threshold of 0.50. The model score has not been calibrated into a verified fraud probability.

### Local Testing

The application was tested with 2,000 transactions.

| Test | Result |
|---|---|
| Transactions scored | 2,000 |
| Review recommendations | 156 |
| Review rate | 7.8% |
| Observed batch scoring time | 0.097 seconds |
| Missing-column validation | Passed |
| Invalid numerical input validation | Passed |

Performance measurements are from local testing and do not include end-to-end deployment latency.

## 7. Repository Structure

```text
Finlora-fraud-risk-scoring/
├── Notebook/
│   ├── data_cleaning_process.ipynb
│   ├── finlora_eda.ipynb
│   ├── finlora_feature_engineering.ipynb
│   └── finlora_model_training.ipynb
├── Streamlit App/
│   └── app.py
├── .gitignore
└── README.md
```

## 8. Running the Streamlit Application Locally

Use a Python environment with the project dependencies, including:

- Streamlit
- NumPy
- Pandas
- SciPy
- Joblib
- scikit-learn 1.7.2

Install the dependencies:

```bash
python -m pip install streamlit numpy pandas scipy joblib scikit-learn==1.7.2
```

The application requires the locally exported Random Forest package at:

```text
Streamlit App/artifacts/finlora_rf_pipeline.joblib
```

This artifact is excluded from the GitHub repository and must be obtained from the project owner or regenerated using the model-training notebook and compatible training artifacts.

To launch the application from the repository root:

```bash
python -m streamlit run "Streamlit App/app.py"
```

The local dashboard is normally available at `http://localhost:8501`.

## 9. Limitations

- The project uses synthetic transaction data.
- Historical spending-baseline construction includes an unresolved fallback for transactions without sufficient prior history.
- The model depends on historical features that must be available and correctly calculated at prediction time.
- The Random Forest scores are not probability-calibrated.
- Model results may change under chronological evaluation and across transaction segments.
- The Streamlit dashboard is a research prototype, not a production financial decision-making system.


## 10. Project Status

**Completed:** Data preparation, feature engineering, baseline model training, model evaluation, initial error analysis, and local Streamlit scoring and review-queue testing.



