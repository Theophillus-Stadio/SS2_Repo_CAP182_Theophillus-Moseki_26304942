# CAP182 Part B - Fraud Detection - Theophillus Moseki 26304942

**Dataset:** Credit Card Fraud - 1,296,675 train / 555,719 test, 0.57% fraud.

This implements two models: XGBoost and Hybrid Isolation Forest + Neural Network.

## Documentation
- [Preprocessing Details](./Preprocessing.MD)
- [Feature Engineering Details](./FeatureEngineering.MD)
- [Model 1 - XGBoost with SMOTE](./Model1.MD)
- [Model 2 - Hybrid Isolation Forest + NN](./Model2.MD)

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