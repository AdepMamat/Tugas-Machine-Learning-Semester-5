import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
df = pd.read_csv("pertemuan_4/processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Data shapes:", X_train.shape, X_val.shape, X_test.shape)

# Function to build model with different configurations
def build_model(neuron1, neuron2, dropout_rate, use_l2=False, use_bn=False):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    if use_l2:
        model.add(layers.Dense(neuron1, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(layers.Dense(neuron1, activation="relu"))
    if use_bn:
        model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    if use_l2:
        model.add(layers.Dense(neuron2, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(layers.Dense(neuron2, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Configurations to test
configs = [
    {"neurons": (32, 16), "dropout": 0.3, "optimizer": "adam", "lr": 1e-3, "l2": False, "bn": False},
    {"neurons": (64, 32), "dropout": 0.3, "optimizer": "adam", "lr": 1e-3, "l2": False, "bn": False},
    {"neurons": (128, 64), "dropout": 0.3, "optimizer": "adam", "lr": 1e-3, "l2": False, "bn": False},
    {"neurons": (32, 16), "dropout": 0.5, "optimizer": "adam", "lr": 1e-3, "l2": True, "bn": False},
    {"neurons": (32, 16), "dropout": 0.3, "optimizer": "sgd", "lr": 0.01, "l2": False, "bn": True},
    {"neurons": (64, 32), "dropout": 0.4, "optimizer": "adam", "lr": 1e-4, "l2": False, "bn": False},
]

results = []

for i, config in enumerate(configs):
    print(f"\n--- Experiment {i+1}: {config} ---")
    model = build_model(config["neurons"][0], config["neurons"][1], config["dropout"], config["l2"], config["bn"])
    
    if config["optimizer"] == "adam":
        opt = keras.optimizers.Adam(config["lr"])
    elif config["optimizer"] == "sgd":
        opt = keras.optimizers.SGD(config["lr"], momentum=0.9)
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    
    # Evaluate
    loss, acc, auc_val = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "config": config,
        "acc": acc,
        "auc": auc_val,
        "f1": f1,
        "history": history.history,
        "y_proba": y_proba,
        "y_pred": y_pred
    })
    
    print(f"Test Acc: {acc:.3f}, AUC: {auc_val:.3f}, F1: {f1:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

# Plot learning curves for all experiments
plt.figure(figsize=(15, 10))
for i, res in enumerate(results):
    plt.subplot(3, 2, i+1)
    plt.plot(res["history"]["loss"], label="Train Loss")
    plt.plot(res["history"]["val_loss"], label="Val Loss")
    plt.title(f"Exp {i+1}: {res['config']['neurons']} neurons, {res['config']['optimizer']}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
plt.tight_layout()
plt.savefig("learning_curves_all.png", dpi=120)
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for i, res in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Exp {i+1} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig("roc_curves.png", dpi=120)
plt.show()

# Threshold analysis for best model (highest F1)
best_idx = np.argmax([r["f1"] for r in results])
best_res = results[best_idx]
thresholds = np.arange(0.1, 0.9, 0.1)
f1_scores = []
for thresh in thresholds:
    y_pred_thresh = (best_res["y_proba"] >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

plt.figure()
plt.plot(thresholds, f1_scores, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title(f'Threshold Analysis for Best Model (Exp {best_idx+1})')
plt.savefig("threshold_analysis.png", dpi=120)
plt.show()

# Final Report
print("\n=== FINAL REPORT ===")
for i, res in enumerate(results):
    print(f"Experiment {i+1}: {res['config']}")
    print(f"  Accuracy: {res['acc']:.3f}, AUC: {res['auc']:.3f}, F1: {res['f1']:.3f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, res['y_pred'])}")
    print()

print("Best Model: Experiment", best_idx+1, "with F1:", results[best_idx]["f1"])
print("Architecture: Input -> Dense(neuron1, relu) -> Dropout -> Dense(neuron2, relu) -> Dense(1, sigmoid)")
print("Chosen for highest F1 score on test set.")
print("Plots saved: learning_curves_all.png, roc_curves.png, threshold_analysis.png")

