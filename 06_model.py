import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import os

os.makedirs('charts_pro', exist_ok=True)

BACKGROUND = '#F8F9FA'
PRIMARY     = '#2C3E50'
ACCENT      = '#E74C3C'
GREEN       = '#27AE60'

df = pd.read_csv('diabetic_data_cleaned.csv')

# ── Step 1: Prepare Features ──────────────────────────────
# Drop columns not useful for prediction
drop_cols = ['encounter_id', 'patient_nbr', 'readmitted',
             'diag_1', 'diag_2', 'diag_3']
df_model = df.drop(columns=drop_cols)

# Encode all categorical columns
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

print("Features prepared ✓")
print(f"Total features: {df_model.shape[1] - 1}")

# ── Step 2: Split Data ────────────────────────────────────
X = df_model.drop('readmitted_binary', axis=1)
y = df_model['readmitted_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Testing samples:  {X_test.shape[0]:,}")

# ── Step 3: Train Model ───────────────────────────────────
print("\nTraining Random Forest model... (this may take 1-2 minutes)")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained successfully ✓")

# ── Step 4: Evaluate Model ────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, y_pred,
      target_names=['Low Risk', 'High Risk']))
print(f"ROC-AUC Score: {auc:.4f}")

# ── Step 5: Confusion Matrix Chart ───────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            linewidths=2, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"}, ax=ax)
ax.set_title('Confusion Matrix — Readmission Prediction',
             fontsize=14, fontweight='bold', color=PRIMARY, pad=15)
ax.set_xlabel('Predicted', fontsize=12, color=PRIMARY)
ax.set_ylabel('Actual', fontsize=12, color=PRIMARY)
plt.tight_layout()
plt.savefig('charts_pro/06_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("\nChart 6 (Confusion Matrix) saved ✓")

# ── Step 6: ROC Curve Chart ───────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5,
        label=f'Random Forest (AUC = {auc:.3f})')
ax.plot([0,1], [0,1], color='#AAAAAA',
        linewidth=1.5, linestyle='--', label='Random Guess')
ax.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
ax.set_title('ROC Curve — Readmission Risk Model',
             fontsize=14, fontweight='bold', color=PRIMARY, pad=15)
ax.set_xlabel('False Positive Rate', fontsize=12, color=PRIMARY)
ax.set_ylabel('True Positive Rate', fontsize=12, color=PRIMARY)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('charts_pro/07_roc_curve.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 7 (ROC Curve) saved ✓")

# ── Step 7: Feature Importance Chart ─────────────────────
feat_imp = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
colors = [ACCENT if i < 5 else '#F0A500' if i < 10 else GREEN
          for i in range(len(feat_imp))]
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
               color=colors[::-1], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, feat_imp.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9,
            fontweight='bold', color=PRIMARY)
ax.set_title('Top 15 Features Driving Readmission Risk',
             fontsize=14, fontweight='bold', color=PRIMARY, pad=15)
ax.set_xlabel('Feature Importance Score', fontsize=12, color=PRIMARY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('charts_pro/08_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 8 (Feature Importance) saved ✓")

print("\n" + "=" * 50)
print("TOP 10 FEATURES DRIVING READMISSION:")
print("=" * 50)
print(feat_imp.head(10).to_string())
print(f"\nModel AUC Score: {auc:.4f}")
print("All model files saved successfully ✓")