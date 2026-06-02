import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

from src.preprocess import preprocess

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/telco_churn.csv')
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True).to_dict()}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
X, y, encoders, scaler = preprocess(df, fit=True)

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Handle Class Imbalance with SMOTE ─────────────────────────────────────
# WHY SMOTE: ~30% churn vs 70% no-churn → model would bias toward majority
# SMOTE creates synthetic minority (churn) samples to balance training data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE - Class distribution: {pd.Series(y_train_sm).value_counts().to_dict()}")

# ── 5. Define Models ──────────────────────────────────────────────────────────
models = {
    # WHY Logistic Regression: simple baseline, highly interpretable
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),

    # WHY Random Forest: ensemble of decision trees, handles non-linearity well
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),

    # WHY XGBoost: gradient boosting, best on tabular data, handles missing values
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric='logloss', verbosity=0),

    # WHY LightGBM: faster than XGBoost, great with categorical features, less memory
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

# ── 6. Train & Evaluate All Models ───────────────────────────────────────────
results = {}
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'model': model, 'accuracy': acc, 'auc': auc,
                     'y_pred': y_pred, 'y_prob': y_prob}

    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ── 7. Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']
print(f"\nBest Model: {best_name} (AUC: {results[best_name]['auc']:.4f})")

# ── 8. Save Best Model ────────────────────────────────────────────────────────
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/best_model_name.pkl', 'wb') as f:
    pickle.dump(best_name, f)
print("Model saved to models/best_model.pkl")

# ── 9. ROC Curve Plot ─────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.close()

# ── 10. Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()

# ── 11. SHAP Explainability ───────────────────────────────────────────────────
# WHY SHAP: tells us WHICH features push a customer toward churning
# and by HOW MUCH — makes ML model interpretable to business stakeholders
print("\nGenerating SHAP explanations...")
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test[:200])

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test[:200], plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved.")
except Exception as e:
    print(f"SHAP skipped: {e}")

# ── 12. Feature Importance Plot ───────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
    fi = fi.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    fi.plot(kind='bar', color='steelblue')
    plt.title(f'Top 15 Feature Importances - {best_name}')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()

print("\nAll plots saved to plots/")
print("Training complete!")
