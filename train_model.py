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
    num_cols = X.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns.tolist()

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
    y_proba = clf.predict_proba(X_test)[:, 1]
    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    try:
        print('\nROC AUC:', roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    # Save pipeline (compressed)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    print("\nCompressing and saving model...")
    joblib.dump(clf, out_path, compress=3)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'✅ Compressed model saved to {out_path} ({size_mb:.2f} MB)')

    # Save columns for app reindexing
    joblib.dump(columns, 'models/columns.pkl')
    print('Saved training columns to models/columns.pkl')

    # Try to extract feature importances and direction
    feature_info = []
    try:
        rf = clf.named_steps['rf']
        importances = rf.feature_importances_
        if len(importances) == len(columns):
            df_imp = pd.DataFrame({'feature': columns, 'importance': importances})
        else:
            df_imp = pd.DataFrame({'feature': columns, 'importance': [0.0] * len(columns)})
            print('Warning: feature_importances length mismatch. Saving zeros for importances.')

        directions = {}
        for col in columns:
            try:
                series = pd.to_numeric(X_train[col], errors='coerce')
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
        df_imp = df_imp.sort_values('importance', ascending=False)
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
