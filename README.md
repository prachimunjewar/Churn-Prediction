# 📉 Customer Churn Prediction System

A machine learning system that predicts which telecom customers are likely to churn, using LightGBM, SMOTE balancing, and SHAP explainability — deployed as an interactive Streamlit web app.
---

## 🎯 Problem Statement

Telecom companies lose significant revenue when customers leave (churn). This system predicts churn risk for individual customers so businesses can take proactive retention action before losing them.
---

## 🚀 Live Demo
https://churn-prediction-gynlsm6nz9gji5angv6zrd.streamlit.app/

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
| Logistic Regression | 67.71% | 0.7432 |
| Random Forest | 69.48% | 0.7433 |
| XGBoost | 66.00% | 0.7224 |
| **LightGBM ✅ (Best)** | **68.28%** | **0.7443** |

---

## 🛠️ Tech Stack

Language - Python 
Data Processing - Pandas, NumPy 
ML Models - Scikit-learn, XGBoost, LightGBM 
Class Imbalance - SMOTE (imbalanced-learn) 
Explainability - SHAP 
Visualization - Matplotlib, Seaborn 
Deployment - Streamlit 
Serialization - Pickle 

---

## 📁 Project Structure

churn_project/
├── src/
│   ├── preprocess.py        # Data cleaning, encoding, scaling
│   ├── train.py             # Model training, evaluation, plots
│   └── predict.py           # Single & batch prediction
├── notebooks/
│   └── EDA_and_modeling.ipynb  # Full EDA + model comparison
├── app.py                   # Streamlit web application
├── requirements.txt         # Dependencies with versions
└── README.md

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/prachimunjewar/customer-churn-prediction
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download **Telco Customer Churn** from Kaggle:
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place it at: `data/telco_churn.csv`

### 4. Train the model
```bash
python -c "import sys; sys.path.insert(0,'.'); exec(open('src/train.py').read())"
```
This will generate trained model files in `models/` and plots in `plots/`

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
---

## 🔍 App Features

- **🔍 Predict Tab** — Enter customer details and get real-time churn probability + risk level
- **📊 Model Performance Tab** — ROC curves, confusion matrix, feature importance for all 4 models
- **🧠 SHAP Insights Tab** — Visual explanation of which features drive churn predictions

---

## 💡 Key Insights from EDA

| Finding | Insight |
| Contract type | Month-to-month customers churn 3x more than two-year contracts |
| Tenure | Churned customers average ~15 months vs ~37 months for loyal customers |
| Monthly charges | Churned customers pay ~$10 more/month on average |
| Internet service | Fiber optic users have highest churn rate |
| Payment method | Electronic check users churn the most |

---

## 🧠 Why These Choices?

**Why SMOTE?**
Dataset has ~31% churn vs 69% no-churn. Without balancing, the model would predict "no churn" always and still get 69% accuracy. SMOTE creates synthetic minority samples to force the model to learn real churn patterns.

**Why LightGBM?**
Uses leaf-wise tree growth (vs XGBoost's level-wise), making it faster and more accurate on this dataset. Achieved highest AUC (0.7443) among all 4 models tested.

**Why SHAP?**
Makes the black-box model interpretable. Business stakeholders can see exactly WHY a customer is predicted to churn — critical for real-world deployment.

---

## 👩‍💻 Author

**Prachi Munjewar**
📧 prachimunjewar@gmail.com

