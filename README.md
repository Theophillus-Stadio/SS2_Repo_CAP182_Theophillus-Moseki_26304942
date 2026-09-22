# CAP182 Part B - Fraud Detection - Theophillus Moseki 26304942

**Dataset:** Credit Card Fraud - 1,296,675 train / 555,719 test, 0.57% fraud.

======= HEAD
This implements two models: XGBoost and Hybrid Isolation Forest + Neural Network.
=======
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
├── Model1Performance.MD          ← Part C: Model 1 results and statistics
├── Model2Performance.MD          ← Part C: Model 2 results and statistics
├── Comparison.MD                 ← Part C: Model 1 vs Model 2 statistical comparison
├── requirements.txt
├── data/
│   ├── raw/                      ← place the downloaded Kaggle CSVs here
│   └── processed/                ← pipeline outputs are written here
├── models/                       ← saved model/scaler artefacts are written here
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── model1_random_forest.py
    ├── model2_hybrid_iforest_nn.py
    ├── metrics_utils.py          ← shared metrics/statistics helpers (Part C)
    ├── model1_performance.py
    ├── model2_performance.py
    └── compare_models.py

reports/                          ← Part C outputs: results JSON, prediction CSVs, plots
```
======= a28e817 (Link Part C performance/comparison docs and add run instructions to README)

## Documentation
- [Preprocessing Details](./Preprocessing.MD)
- [Feature Engineering Details](./FeatureEngineering.MD)
- [Model 1 - XGBoost with SMOTE](./Model1.MD)
- [Model 2 - Hybrid Isolation Forest + NN](./Model2.MD)

======= HEAD
## Repository Structure
fraud-detection-part-b/
├── README.md
├── requirements.txt
├── data/raw/
├── src/
├── notebooks/
├── models/

## How to Run
pip install -r requirements.txt
=======
| Stage | Details | Script(s) |
|---|---|---|
| Data preprocessing | [Preprocessing.MD](Preprocessing.MD) | `src/preprocessing.py` |
| Feature engineering | [FeatureEngineering.MD](FeatureEngineering.MD) | `src/feature_engineering.py` |
| Model 1 — Random Forest | [Model1.MD](Model1.MD) | `src/model1_random_forest.py` |
| Model 2 — Isolation Forest + Neural Network | [Model2.MD](Model2.MD) | `src/model2_hybrid_iforest_nn.py` |
| Model 1 performance results | [Model1Performance.MD](Model1Performance.MD) | `src/model1_performance.py` |
| Model 2 performance results | [Model2Performance.MD](Model2Performance.MD) | `src/model2_performance.py` |
| Model 1 vs Model 2 comparison | [Comparison.MD](Comparison.MD) | `src/compare_models.py` |

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

## Evaluating and comparing the two models (Part C)

```bash
# 4a. Model 1 results (metrics + bootstrap confidence intervals)
python src/model1_performance.py \
    --model models/model1_random_forest.joblib \
    --test data/processed/test_features.csv \
    --output-dir reports

# 4b. Model 2 results (metrics + bootstrap confidence intervals)
python src/model2_performance.py \
    --iso-model models/model2_isolation_forest.joblib \
    --nn-model models/model2_neural_network.keras \
    --test data/processed/test_features.csv \
    --output-dir reports

# 4c. Statistical comparison (McNemar's test + paired bootstrap)
python src/compare_models.py \
    --model1-predictions reports/model1_predictions.csv \
    --model2-predictions reports/model2_predictions.csv \
    --output-dir reports
```

See each linked `.MD` file for the reasoning behind each stage's design
choices, tied back to the sources reviewed in Part A.
======= a28e817 (Link Part C performance/comparison docs and add run instructions to README)
