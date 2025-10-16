import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

print("Distribusi label awal:")
print(y.value_counts())

# === Preprocessing (numerik saja) ===
num_cols = X.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

# === Baseline Model: Logistic Regression ===
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

# Cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores_lr = cross_val_score(pipe_lr, X, y, cv=skf, scoring="f1_macro")

print("\nBaseline Logistic Regression (CV F1 scores):", scores_lr)
print("Mean F1 (LogReg):", scores_lr.mean())

# === Alternative Model: Random Forest ===
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

scores_rf = cross_val_score(pipe_rf, X, y, cv=skf, scoring="f1_macro")
print("\nRandomForest (CV F1 scores):", scores_rf)
print("Mean F1 (RF):", scores_rf.mean())

# === Hyperparameter tuning (GridSearch) ===
param = {
  "clf__max_depth": [None, 5, 10],
  "clf__min_samples_split": [2, 5]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X, y)

print("\nBest params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

# === Evaluasi akhir (pakai model terbaik full data) ===
best_model = gs.best_estimator_
best_model.fit(X, y)

# Kalau ada probabilitas, tampilkan ROC
if hasattr(best_model, "predict_proba") and len(y.unique()) == 2:
    y_proba = best_model.predict_proba(X)[:, 1]
    print("ROC-AUC (train on full data):", roc_auc_score(y, y_proba))

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Full Data)")
    plt.legend()
    plt.show()
