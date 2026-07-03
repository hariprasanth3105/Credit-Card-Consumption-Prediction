# Predicting Credit Card Consumption for a Bank

A capstone machine learning project that predicts customers' average credit card spend for the next three months (July–September) using their demographics, transaction history, and banking behaviour from the preceding quarter (April–June).

## Business Context

Credit card issuers sit on a rich stream of behavioural data — spending patterns, transaction frequency, loan activity, investments — that can be leveraged to understand and anticipate customer needs. Accurately forecasting future credit card spend helps a bank plan credit limits, design targeted marketing campaigns, and manage customer relationships more effectively.

A leading bank provided three months of customer demographic and behavioural data, along with historical credit consumption figures, but a portion of customers had missing target values. The objective of this project is to build a regression model on the customers with known consumption and use it to predict consumption for the customers with missing values.

## Objective

Given:
- **Customer demographics** (age, income, tenure, account type, region, etc.)
- **Customer behavioural data** (credit/debit card spend and transaction counts for Apr–Jun, credit limits, loans, investments, EMIs, etc.)
- **Historical credit consumption** (`cc_cons`), with some customers missing this value

Build a regression model to predict `cc_cons` (average credit card spend over the next three months) for the customers with missing target values, and validate the model using **Root Mean Square Percentage Error (RMSPE)**.

## Dataset

| File | Description |
|---|---|
| `CustomerDemographics.csv` | Customer ID, account type, gender, age, income level, employment tenure, tenure with bank, region code, net banking flag, average days between transactions |
| `CustomerBehaviorData.csv` | Monthly credit/debit card spend and transaction counts (Apr–Jun), card limit, active/closed personal & vehicle loans, investment flags (DEMAT, FD, life & general insurance), monthly debit/credit amounts and counts, loan enquiry flag, active EMI |
| `CreditConsumptionData.csv` | Customer ID and `cc_cons` — the target variable (average credit card spend for the next 3 months). Missing for a subset of customers, which is what the model predicts |

- **Total customers:** 20,000
  - **Training set** (target known): 15,000
  - **Prediction set** (target missing): 5,000

## Approach

### 1. Data Preparation
- Merged demographics, behavioural, and target data on customer `ID`.
- Imputed missing numeric features with the **median** and missing categorical features with the **mode**.
- Treated outliers in continuous features using the **IQR method** (clipped values beyond 1.5×IQR from Q1/Q3), leaving binary/flag columns untouched.

### 2. Feature Engineering
Engineered additional features to better capture spending behaviour, including:
- Total and average credit card spend over the last 3 months
- Month-over-month CC spend growth (Apr→May, May→Jun)
- Total debit card spend and CC-to-DC spend ratio
- Total transaction count across credit and debit cards
- Aggregated active/closed loan counts and total loans
- Total investment across all investment types
- EMI-active flag
- Age bands (Young / Adult / Middle / Senior / Elder)

Categorical variables (gender, account type, income, region code, age group, loan enquiry) were **one-hot encoded**.

### 3. Feature Selection
Used **Recursive Feature Elimination (RFE)** with a Linear Regression estimator to shortlist the top 40 most predictive features, reducing noise and dimensionality before model training.

### 4. Modelling
Trained and compared multiple **XGBoost Regressor** configurations:

| Model | Description | Train RMSPE | Test RMSPE |
|---|---|---|---|
| M1 | Baseline XGBoost (default params) | 21.25% | 42.95% |
| M2 | Hyperparameter-tuned via `GridSearchCV` (5-fold CV) | 35.05% | 36.83% |
| **M3** | Tuned XGBoost trained on **log1p-transformed target** | **3.20%** | **3.34%** |

- Hyperparameter tuning searched over `n_estimators`, `max_depth`, and `learning_rate`, optimizing for RMSE via cross-validation.
- The target variable (`cc_cons`) was highly right-skewed, which limited the model's ability to predict high-spend customers accurately. Applying a **log1p transformation** to the target dramatically stabilized the error metric and improved generalization — RMSPE dropped from ~37% to ~3.4% on the test set.
- `log1p` (rather than plain `log`) was used to safely handle customers with zero recorded spend.

### 5. Final Prediction
- Retrained the best model (M3's configuration) on the **full training set** (all 15,000 customers with known targets) for maximum data utilization.
- Generated predictions on the 5,000 customers with missing `cc_cons`, then reversed the log transformation using `expm1` to return predictions to the original spend scale.
- Final predictions exported to `Final_Prediction.csv`.

## Model Evaluation Metric

Root Mean Square Percentage Error (RMSPE):

```
RMSPE = sqrt( mean( ((y_true - y_pred) / y_true)^2 ) ) * 100
```

## Results

- Final model (XGBoost + log-transformed target) achieved an RMSPE of **~3.2%** on the full training data.
- R² improved from ~0.48 (baseline, untransformed target) to **~0.78** on the held-out test set after the log transformation.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn (train_test_split, StandardScaler, RFE, GridSearchCV, metrics), XGBoost

## Repository Structure

```
├── CreditCard_CaseStudy.ipynb        # Full analysis: EDA, feature engineering, modelling, prediction
├── Final_Prediction.csv              # Predicted cc_cons for customers with missing target values
└── README.md
```

## Key Learnings

- Target transformation (log1p) can be far more impactful than hyperparameter tuning alone when the target distribution is heavily skewed.
- RFE-based feature selection helped simplify the model without sacrificing performance.
- IQR-based outlier clipping (applied selectively, avoiding binary/flag columns) improved robustness of the downstream model.

---
