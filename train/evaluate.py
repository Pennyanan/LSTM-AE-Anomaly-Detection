from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    classification_report
)

# =========================
# 路徑設定
# =========================
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

WINDOW_SIZE = 20
STRIDE = 1
FAULT_START = 160


# =========================
# 前處理
# =========================
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # 建 sample
    if "sample" not in df.columns:
        df["sample"] = np.arange(len(df))

    # 建 run
    if "simulationRun" not in df.columns:
        df["simulationRun"] = 0

    # 建 label
    df["label"] = ((df["faultNumber"] > 0) & (df["sample"] >= FAULT_START)).astype(int)

    return df


def sliding_window(df, feature_cols):
    X, y = [], []

    data = df[feature_cols].values
    labels = df["label"].values

    for i in range(len(df) - WINDOW_SIZE + 1):
        X.append(data[i:i + WINDOW_SIZE])
        y.append(int(labels[i:i + WINDOW_SIZE].max()))

    return np.array(X), np.array(y)


# =========================
# 主程式
# =========================
def main(file_path):

    print("[INFO] Loading artifacts...")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    print(f"[INFO] Threshold: {threshold:.6f}")

    print("[INFO] Loading data...")
    df = pd.read_csv(file_path)

    df = preprocess(df)

    # feature 對齊
    df = df[feature_cols + ["label"]]

    # scaling
    df[feature_cols] = scaler.transform(df[feature_cols])

    print("[INFO] Sliding window...")
    X, y_true = sliding_window(df, feature_cols)

    print(f"[INFO] Num windows: {len(X)}")

    print("[INFO] Predicting...")
    X_pred = model.predict(X, verbose=1)
    errors = np.mean(np.abs(X_pred - X), axis=(1, 2))

    y_pred = (errors > threshold).astype(int)

    # =========================
    # Metrics
    # =========================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fdr = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, errors)
    pr_auc = auc(pr_recall, pr_precision)

    print("\n===== Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"FDR      : {fdr:.4f}")
    print(f"FAR      : {far:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"PR AUC   : {pr_auc:.4f}")

    # =========================
    # Save outputs
    # =========================
    results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "FDR": fdr,
        "FAR": far,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }

    with open(OUTPUT_DIR / "quick_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close()

    # PR
    plt.figure()
    plt.plot(pr_recall, pr_precision, label=f"AUC={pr_auc:.4f}")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "pr_curve.png")
    plt.close()

    print(f"\n[INFO] Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    main(args.file)
