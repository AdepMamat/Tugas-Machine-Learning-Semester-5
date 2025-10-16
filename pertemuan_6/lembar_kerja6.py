# lembar_kerja_fix.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import joblib

# --- 0. Optional: cek working dir ---
# print("Working dir:", os.getcwd())

# --- 1. Load data ---
df = pd.read_csv("processed_kelulusan.csv")  # pastikan file ada di folder yg sama

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# --- 2. Split: 70/15/15 ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# --- 3. Preprocessing & pipeline ---
num_cols = X_train.select_dtypes(include="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])

# --- 4. Fit baseline RF ---
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# --- 5. Cross validation (train) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train):", scores.mean(), "±", scores.std())

# --- 6. GridSearchCV tuning ---
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# --- 7. Final evaluation on test set ---
final_model = best_model
y_test_pred = final_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# --- 8. ROC & PR curves (if predict_proba exists) ---
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except Exception as e:
        print("Gagal hitung ROC-AUC:", e)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    plt.legend()
    plt.tight_layout(); plt.savefig("roc_test.png", dpi=120); plt.close()

    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (test)")
    plt.legend()
    plt.tight_layout(); plt.savefig("pr_test.png", dpi=120); plt.close()

# --- 9. Feature importance (robust) ---
try:
    importances = final_model.named_steps["clf"].feature_importances_
    # try to get feature names produced by preprocessor
    try:
        feat_names = final_model.named_steps["pre"].get_feature_names_out()
    except Exception:
        # fallback: use numeric columns as names (in same order)
        feat_names = np.array(num_cols)
    top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print("Top feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# --- 10. Simpan model ---
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl")

# --- 11. Cek inference lokal (contoh) ---
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])[num_cols]  # pastikan urutan kolom sama dengan num_cols
print("Prediksi contoh (0=tidak lulus, 1=lulus):", int(mdl.predict(sample)[0]))
