from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
MODEL_PATH = 'models/model.pkl'
COLUMNS_PATH = 'models/columns.pkl'
FEATURE_INFO_PATH = 'models/feature_info.pkl'

# Load model and helpers at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train and save model first.")

model = joblib.load(MODEL_PATH)

# Load expected columns if available
expected_cols = None
if os.path.exists(COLUMNS_PATH):
    expected_cols = joblib.load(COLUMNS_PATH)
else:
    # As a fallback, try to infer from model if possible (not robust)
    expected_cols = None
    print("Warning: models/columns.pkl not found - app will accept whatever fields are sent and may error.")

# Load feature info (if present)
feature_info = None
if os.path.exists(FEATURE_INFO_PATH):
    try:
        feature_info = joblib.load(FEATURE_INFO_PATH)
    except Exception:
        feature_info = None

def preprocess_input_dict(d):
    # convert to DataFrame and reindex to expected order, filling missing with NaN
    df = pd.DataFrame([d])
    # Try numeric conversion
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
    if expected_cols is not None:
        # reindex to expected columns
        df = df.reindex(columns=expected_cols)
    return df

def rule_based_explanation(input_row):
    # fallback explanation when feature_info missing: simple rules
    ex = []
    try:
        r = input_row.iloc[0]
        # example heuristics:
        if 'PAY_0' in r.index and pd.notna(r['PAY_0']) and r['PAY_0'] > 0:
            ex.append("Client had recent late payment(s) — increases default risk.")
        if 'LIMIT_BAL' in r.index and pd.notna(r['LIMIT_BAL']) and r['LIMIT_BAL'] < 50000:
            ex.append("Low credit limit compared to population — may increase risk.")
        # high recent bill amount relative to payment may indicate stress
        if all(x in r.index for x in ['BILL_AMT1','PAY_AMT1']) and pd.notna(r['BILL_AMT1']) and pd.notna(r['PAY_AMT1']):
            if r['PAY_AMT1'] < 0.1 * (r['BILL_AMT1'] if r['BILL_AMT1']>0 else 1):
                ex.append("Recent payment covers small portion of the bill — potential risk.")
    except Exception:
        pass
    if not ex:
        ex = ["No strong simple-rule indicators found in inputs."]
    return " ".join(ex)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form.to_dict()
    # Convert string "null" or None gracefully
    # Build dataframe
    try:
        row = preprocess_input_dict(data)
        # predict
        proba = None
        pred = None
        try:
            proba = float(model.predict_proba(row)[:,1][0])
            pred = int(model.predict(row)[0])
        except Exception as e:
            # If model fails (e.g. missing columns), return helpful error
            return jsonify({'error': f'Prediction failed: {str(e)}'})

        # Build explanation using saved feature_info if available
        explanation = ""
        top_features = []
        if feature_info:
            # feature_info is list of dicts: feature, importance, corr, direction
            # include only top 5
            top_features = feature_info[:5]
            # build textual explanation
            parts = []
            for f in top_features:
                feat = f.get('feature')
                imp_pct = round(f.get('importance',0)*100,2)
                direction = f.get('direction','no clear direction')
                parts.append(f"{feat} ({imp_pct}% importance, {direction})")
            explanation = "Top contributors: " + "; ".join(parts)
        else:
            explanation = rule_based_explanation(row)

        # also include a compact list of top features (for frontend)
        response = {
            'prediction': pred,
            'probability': proba,
            'explanation': explanation,
            'feature_importances': top_features
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# allow serving placeholders from static (optional)
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'), filename)

if __name__ == '__main__':
    # Flask debug for local use
    app.run(host='0.0.0.0', port=5000, debug=True)
