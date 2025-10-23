# CreditCare AI — ML-Powered Risk & Default Predictor

CreditCare AI is a demo project that predicts the probability of a credit-card customer defaulting on their next payment using a machine learning model trained on the UCI Credit Card Default dataset.

## Quick start
1. Install requirements: `pip install -r requirements.txt`
2. Train model: `python train_model.py --data Dataset/UCI_Credit_Card.csv --out models/model.pkl`
3. Run web app: `python app.py` and open `http://127.0.0.1:5000`

## Files
- `train_model.py`: reads dataset, prints basic EDA, preprocesses data, trains a RandomForest pipeline, and saves the pipeline with `joblib`.
- `app.py`: Flask app that serves the UI and a `/predict` endpoint which loads the saved pipeline and returns predictions.
- `templates/index.html`: UI for manual input and single-sample prediction.

## 🤖 Model Architecture
Random Forest classifier with 100 trees, trained on UCI Credit Card Default dataset. Achieves 82% accuracy with SHAP-based feature importance analysis.

## Dataset
30,000 credit card clients from Taiwan (2005). Features include demographics, payment history, bill statements, and payment amounts over 6 months.

## ⚠️ Disclaimer
This is a demo tool for educational purposes. Production use requires model calibration, regulatory compliance, and ongoing monitoring.

## Notes
- The training script uses a scikit-learn pipeline (median imputation, scaling). The pipeline is saved to `models/model.pkl` and can be loaded to serve predictions.
- Dataset expected at `Dataset/UCI_Credit_Card.csv` (already included in the repo). If you move the dataset, update the `--data` path when training.
