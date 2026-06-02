import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def preprocess(df, fit=True, encoders=None, scaler=None):
    """
    Clean and encode the Telco churn dataset.
    
    WHY THIS APPROACH:
    - Label Encoding for binary columns (Yes/No) → simple 0/1
    - Label Encoding for multi-class categoricals → works well with tree models
    - StandardScaler for numerical cols → needed for Logistic Regression
    - Drop customerID → no predictive value
    """

    df = df.copy()

    # Drop ID column - not a feature
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # Convert TotalCharges to numeric (sometimes comes as string)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Identify categorical columns
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if fit:
        # Fit new encoders
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # Scale numerical features
        # WHY: Logistic Regression is sensitive to feature scale
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Save encoders and scaler for later use in app
        with open('models/encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Use existing encoders for prediction
        for col in cat_cols:
            X[col] = encoders[col].transform(X[col])
        X[num_cols] = scaler.transform(X[num_cols])

    return X, y, encoders, scaler
