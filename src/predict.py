import pandas as pd
import numpy as np
import pickle
import sys
import os

# ── Load Artifacts ────────────────────────────────────────────────────────────
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/best_model_name.pkl', 'rb') as f:
        model_name = pickle.load(f)
    return model, encoders, scaler, model_name


# ── Preprocess Input ──────────────────────────────────────────────────────────
def preprocess_input(df, encoders, scaler):
    """
    Apply same preprocessing as training pipeline.
    WHY: Model was trained on encoded/scaled data —
         prediction input must match exact same format.
    """
    df = df.copy()

    # Drop ID if present
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # Fix TotalCharges dtype
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop target if present (batch prediction on labeled data)
    df.drop(columns=['Churn'], inplace=True, errors='ignore')

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Apply saved encoders
    for col in cat_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # Apply saved scaler
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ── Single Customer Prediction ────────────────────────────────────────────────
def predict_single(customer_dict):
    """
    Predict churn for a single customer.

    Args:
        customer_dict: dict with customer feature values

    Returns:
        dict with prediction, probability and risk level
    """
    model, encoders, scaler, model_name = load_artifacts()

    df = pd.DataFrame([customer_dict])
    df = preprocess_input(df, encoders, scaler)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    risk = (
        "High Risk 🔴"   if prob > 0.70 else
        "Medium Risk 🟡" if prob > 0.40 else
        "Low Risk 🟢"
    )

    return {
        "model_used"        : model_name,
        "churn_prediction"  : "Will Churn" if pred == 1 else "Will Not Churn",
        "churn_probability" : round(float(prob), 4),
        "churn_percent"     : f"{prob*100:.1f}%",
        "risk_level"        : risk
    }


# ── Batch Prediction ──────────────────────────────────────────────────────────
def predict_batch(csv_path, output_path="predictions.csv"):
    """
    Predict churn for multiple customers from a CSV file.

    Args:
        csv_path   : path to input CSV file
        output_path: where to save predictions

    Returns:
        DataFrame with original data + predictions appended
    """
    model, encoders, scaler, model_name = load_artifacts()

    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} records from {csv_path}")

    df_processed = preprocess_input(df_raw.copy(), encoders, scaler)

    probs = model.predict_proba(df_processed)[:, 1]
    preds = (probs > 0.5).astype(int)

    df_raw['Churn_Probability'] = np.round(probs, 4)
    df_raw['Churn_Prediction']  = np.where(preds == 1, 'Will Churn', 'Will Not Churn')
    df_raw['Risk_Level']        = pd.cut(
        probs,
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    df_raw.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Churn breakdown:\n{df_raw['Churn_Prediction'].value_counts()}")

    return df_raw


# ── CLI Usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Usage:
      Single : python src/predict.py single
      Batch  : python src/predict.py batch data/telco_churn.csv
    """

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        # Example customer — month-to-month, short tenure, fiber optic
        # (high churn risk profile based on SHAP insights)
        sample_customer = {
            'gender'          : 'Female',
            'SeniorCitizen'   : 0,
            'Partner'         : 'No',
            'Dependents'      : 'No',
            'tenure'          : 3,
            'PhoneService'    : 'Yes',
            'MultipleLines'   : 'No',
            'InternetService' : 'Fiber optic',
            'OnlineSecurity'  : 'No',
            'OnlineBackup'    : 'No',
            'DeviceProtection': 'No',
            'TechSupport'     : 'No',
            'StreamingTV'     : 'Yes',
            'StreamingMovies' : 'Yes',
            'Contract'        : 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod'   : 'Electronic check',
            'MonthlyCharges'  : 95.5,
            'TotalCharges'    : 286.5
        }

        result = predict_single(sample_customer)
        print("\n── Single Customer Prediction ──────────────────")
        for k, v in result.items():
            print(f"  {k:<22}: {v}")

    elif mode == "batch":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else "data/telco_churn.csv"
        predict_batch(csv_path, output_path="predictions.csv")

    else:
        print("Usage: python src/predict.py [single|batch] [csv_path]")
