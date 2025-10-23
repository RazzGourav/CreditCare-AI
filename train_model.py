


# """Train a classification model on the UCI Credit Card dataset and save a pipeline with the model.

# Usage example:
# python train_model.py --data Dataset/UCI_Credit_Card.csv --out models/model.pkl
# """
# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import joblib




# def load_data(path):
# 	"""Load CSV data into a DataFrame."""
# 	df = pd.read_csv(path)
# 	return df




# def basic_eda(df):
# 	print("Shape:", df.shape)
# 	print("Columns:", df.columns.tolist())
# 	print("Missing values:\n", df.isnull().sum())
# 	if 'default.payment.next.month' in df.columns:
# 		print("Target distribution:\n", df['default.payment.next.month'].value_counts(normalize=True))




# def preprocess_and_train(df, out_path):
# 	"""Preprocess data, train a RandomForest pipeline, and save it to out_path."""
# 	# Rename target for convenience
# 	target_col = 'default.payment.next.month'

# 	# Drop ID if present
# 	if 'ID' in df.columns:
# 		df = df.drop(columns=['ID'])

# 	if target_col not in df.columns:
# 		raise ValueError(f"Target column '{target_col}' not found in dataframe")

# 	X = df.drop(columns=[target_col])
# 	y = df[target_col]

# 	# Numerical: continuous/ordinal numeric features
# 	num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# 	cat_cols = [c for c in X.columns if c not in num_cols]

# 	# Build transformers
# 	num_pipeline = Pipeline([
# 		('imputer', SimpleImputer(strategy='median')),
# 		('scaler', StandardScaler())
# 	])

# 	preprocessor = ColumnTransformer(transformers=[
# 		('num', num_pipeline, num_cols),
# 	], remainder='passthrough')

# 	# Full pipeline
# 	pipe = Pipeline([
# 		('pre', preprocessor),
# 		('clf', RandomForestClassifier(n_estimators=100, random_state=42))
# 	])

# 	# Train/test split
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 	print("Training model...")
# 	pipe.fit(X_train, y_train)

# 	# Evaluate
# 	preds = pipe.predict(X_test)
# 	probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
# 	print("Classification report:\n", classification_report(y_test, preds))
# 	print("Confusion matrix:\n", confusion_matrix(y_test, preds))
# 	if probs is not None:
# 		try:
# 			auc = roc_auc_score(y_test, probs)
# 			print(f"ROC AUC: {auc:.4f}")
# 		except Exception:
# 			pass

# 	# Save pipeline
# 	joblib.dump(pipe, out_path)
# 	print(f"Saved trained pipeline to {out_path}")


# def _parse_args():
# 	parser = argparse.ArgumentParser(description='Train credit card default model')
# 	parser.add_argument('--data', required=True, help='Path to CSV data file')
# 	parser.add_argument('--out', required=True, help='Path to save trained model (joblib .pkl)')
# 	return parser.parse_args()


# def main():
# 	args = _parse_args()
# 	df = load_data(args.data)
# 	basic_eda(df)
# 	preprocess_and_train(df, args.out)


# if __name__ == '__main__':
# 	main()

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from scipy.stats import spearmanr

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_eda(df):
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Target distribution:\n", df['default.payment.next.month'].value_counts(normalize=True))

def preprocess_and_train(df, out_path):
    target_col = 'default.payment.next.month'

    # Drop ID if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Save columns (original order used for reindexing in app)
    columns = X.columns.tolist()

    # Numerical columns
    num_cols = X.select_dtypes(include=['int64','float64','float32','int32']).columns.tolist()

    # Build transformers
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
    ], remainder='passthrough')  # passthrough other columns if any

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Full pipeline with classifier
    clf = Pipeline([
        ('preproc', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print('Training model...')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    try:
        print('\nROC AUC:', roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    # Save pipeline
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(clf, out_path)
    print(f'Pipeline + model saved to {out_path}')

    # Save columns for app reindexing
    joblib.dump(columns, 'models/columns.pkl')
    print('Saved training columns to models/columns.pkl')

    # Try to extract feature importances and direction
    feature_info = []
    try:
        rf = clf.named_steps['rf']
        importances = rf.feature_importances_
        # We assume the number of importances equals number of original columns.
        # If mismatch (due to transformers), attempt to map using columns list.
        if len(importances) == len(columns):
            df_imp = pd.DataFrame({'feature': columns, 'importance': importances})
        else:
            # fallback: try to use columns but warn user
            df_imp = pd.DataFrame({'feature': columns, 'importance': [0.0]*len(columns)})
            print('Warning: feature_importances length mismatch. Saving zeros for importances.')
        # compute simple Spearman correlation as direction (on training data)
        directions = {}
        for col in columns:
            try:
                # numeric conversion
                series = pd.to_numeric(X_train[col], errors='coerce')
                # compute spearman; if too many nans, set 0
                if series.isna().all():
                    corr = 0.0
                else:
                    corr, _ = spearmanr(series.fillna(series.median()), y_train)
                    if np.isnan(corr):
                        corr = 0.0
                directions[col] = float(corr)
            except Exception:
                directions[col] = 0.0

        df_imp['direction_corr'] = df_imp['feature'].map(directions)
        # convert to sorted list
        df_imp = df_imp.sort_values('importance', ascending=False)
        # build feature_info top 10
        for _, row in df_imp.head(10).iterrows():
            feat = row['feature']
            imp = float(row['importance'])
            corr = float(row['direction_corr'])
            direction = 'higher increases risk' if corr > 0.03 else ('higher reduces risk' if corr < -0.03 else 'no clear linear direction')
            feature_info.append({'feature': feat, 'importance': imp, 'corr': corr, 'direction': direction})
    except Exception as e:
        print('Could not compute feature importances:', e)
        feature_info = []

    # Save feature_info
    joblib.dump(feature_info, 'models/feature_info.pkl')
    print('Saved feature_info to models/feature_info.pkl (top features + direction)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to UCI_Credit_Card.csv')
    parser.add_argument('--out', default='models/model.pkl', help='Where to save the pipeline')
    args = parser.parse_args()

    df = load_data(args.data)
    basic_eda(df)
    preprocess_and_train(df, args.out)
