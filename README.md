# STADIOalot Fraud Detection — Part B (SS2)

This repository implements a preprocessing → feature engineering → modelling
pipeline for detecting fraudulent e-commerce transactions, as chosen and
justified in Part A.

**Dataset:** "Credit Card Transactions Fraud Detection Dataset" (Shenoy, 2020),
synthetically generated via Sparkov Data Generation. ~1.85 million transactions,
23 original features, binary target (`is_fraud`), ~0.57% fraud rate.
Available at: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Repository structure

```
stadioalot-fraud-detection/
├── README.md                    ← you are here
├── Preprocessing.MD              ← Part B: preprocessing details
├── FeatureEngineering.MD         ← Part B: feature engineering details
├── Model1.MD                     ← Part B: Model 1 (Random Forest) details
├── Model2.MD                     ← Part B: Model 2 (Isolation Forest + NN) details
├── requirements.txt
├── data/
│   ├── raw/                      ← place the downloaded Kaggle CSVs here
│   └── processed/                ← pipeline outputs are written here
├── models/                       ← saved model/scaler artefacts are written here
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── model1_random_forest.py
    └── model2_hybrid_iforest_nn.py
```

## Documentation

| Stage | Details | Script(s) |
|---|---|---|
| Data preprocessing | [Preprocessing.MD](Preprocessing.MD) | `src/preprocessing.py` |
| Feature engineering | [FeatureEngineering.MD](FeatureEngineering.MD) | `src/feature_engineering.py` |
| Model 1 — Random Forest | [Model1.MD](Model1.MD) | `src/model1_random_forest.py` |
| Model 2 — Isolation Forest + Neural Network | [Model2.MD](Model2.MD) | `src/model2_hybrid_iforest_nn.py` |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `fraudTrain.csv` and `fraudTest.csv` from the Kaggle link above and
place them in `data/raw/`.

## Running the full pipeline end to end

```bash
# 1. Preprocessing (cleaning, encoding, scaling)
python src/preprocessing.py \
    --train data/raw/fraudTrain.csv \
    --test data/raw/fraudTest.csv \
    --output-dir data/processed \
    --artifacts-dir models

# 2. Feature engineering (temporal, contextual, behavioural-velocity features)
python src/feature_engineering.py \
    --train data/processed/train_clean.csv \
    --test data/processed/test_clean.csv \
    --output-dir data/processed

# 3a. Model 1 — Random Forest baseline
python src/model1_random_forest.py \
    --train data/processed/train_features.csv \
    --test data/processed/test_features.csv \
    --output-model models/model1_random_forest.joblib \
    --use-smote

# 3b. Model 2 — Hybrid Isolation Forest + Neural Network
python src/model2_hybrid_iforest_nn.py \
    --train data/processed/train_features.csv \
    --test data/processed/test_features.csv \
    --output-dir models
```

See each linked `.MD` file for the reasoning behind each stage's design
choices, tied back to the sources reviewed in Part A.
