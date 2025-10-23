# Credit-Card-Pulse
Credit Card Default Prediction Objective: Predict if a customer will default on their next payment. Key Concepts: Classification, feature encoding, confusion matrix.

# Credit Card Default Prediction


Predict whether a credit card client will default next month using the UCI dataset.


## Steps
1. Install requirements: `pip install -r requirements.txt`
2. Train model: `python train_model.py --data /mnt/data/UCI_Credit_Card.csv --out models/model.pkl`
3. Run web app: `python app.py` and open `http://127.0.0.1:5000`


## Files
- `train_model.py`: reads dataset, EDA prints, preprocessing, trains model (RandomForest), saves model and preprocessing pipeline with joblib.
- `app.py`: Flask endpoint `/predict` that loads pipeline+model, accepts JSON or form POST and returns prediction and probabilities.
- `templates/index.html`: UI for manual input and single-sample prediction.


## Notes
- The training script uses standard sklearn pipeline (imputation, scaling, one-hot where needed). It also prints confusion matrix and classification report.
- Dataset expected at the path you uploaded: `/mnt/data/UCI_Credit_Card.csv`.
