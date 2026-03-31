from pathlib import Path
from typing import List, Dict, Tuple
import json
import time

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
)

import matplotlib.pyplot as plt


# =========================
# 專案路徑設定
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"


# =========================
# 參數設定
# =========================
T = 960
FAULT_START = 160
WINDOW_SIZE = 20
STRIDE = 1


# =========================
# 基本工具
# =========================
def set_label_based_on_sample(df: pd.DataFrame, onset: int = 160) -> pd.DataFrame:
    d = df.copy()
    d["label"] = ((d["faultNumber"] > 0) & (d["sample"] >= onset)).astype(int)
    return d


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("xmeas_") or col.startswith("xmv_")]


# =========================
# 與 train.py 一致的資料流程
# =========================
def load_and_split_data(
    normal_path: Path,
    faulty_path: Path,
    t: int = 960,
    fault_start: int = 160
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_path)
    df_faulty = pd.read_csv(faulty_path)

    n_run_normal = len(df_normal) // t
    df_normal = df_normal.iloc[:n_run_normal * t].copy()
    df_normal["simulationRun"] = np.repeat(np.arange(n_run_normal), t)
    df_normal["sample"] = np.tile(np.arange(t), n_run_normal)
    df_normal["faultNumber"] = 0
    df_normal["label"] = 0

    n_run_faulty = len(df_faulty) // t
    df_faulty = df_faulty.iloc[:n_run_faulty * t].copy()
    df_faulty["simulationRun"] = np.repeat(np.arange(n_run_faulty), t)
    df_faulty["sample"] = np.tile(np.arange(t), n_run_faulty)
    df_faulty = set_label_based_on_sample(df_faulty, onset=fault_start)

    train_df = df_normal.copy()

    val_list = []
    test_list = []

    for f in sorted(df_faulty["faultNumber"].unique()):
        if f == 0:
            continue

        df_f = df_faulty[df_faulty["faultNumber"] == f].copy()
        run_ids = df_f["simulationRun"].unique()
        n_val = int(len(run_ids) * 0.3)

        val_ids = run_ids[:n_val]
        test_ids = run_ids[n_val:]

        val_list.append(df_f[df_f["simulationRun"].isin(val_ids)])
        test_list.append(df_f[df_f["simulationRun"].isin(test_ids)])

    val_df = pd.concat(val_list, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).reset_index(drop=True)

    val_df.loc[val_df["sample"] < fault_start, "label"] = 0
    test_df.loc[test_df["sample"] < fault_start, "label"] = 0

    return train_df, val_df, test_df


def remove_dropped_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep_cols = [c for c in train_df.columns if c not in get_feature_columns(train_df)] + feature_cols
    train_df = train_df[keep_cols].copy()
    val_df = val_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()
    return train_df, val_df, test_df


def apply_saved_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, val_df, test_df


def sliding_window(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 20,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    X, y, run_ids, sample_ids = [], [], [], []

    for run_id, group in df.groupby("simulationRun", sort=False):
        group = group.sort_values("sample")

        data = group[feature_cols].values
        labels = group["label"].values
        samples = group["sample"].values

        total_len = len(group)
        for i in range(0, total_len - window_size + 1, stride):
            X.append(data[i:i + window_size])
            y.append(int(labels[i:i + window_size].max()))
            run_ids.append(int(run_id))
            sample_ids.append(int(samples[i]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), run_ids, sample_ids


# =========================
# 評估工具
# =========================
def compute_reconstruction_mae(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    X_pred = model.predict(X, verbose=1)
    mae_loss = np.mean(np.abs(X_pred - X), axis=(1, 2))
    return mae_loss


def adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_pa = y_pred.copy()

    in_anomaly = False
    start = 0

    for i, label in enumerate(y_true):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            end = i
            if np.sum(y_pred[start:end]) > 0:
                y_pred_pa[start:end] = 1

    if in_anomaly:
        end = len(y_true)
        if np.sum(y_pred[start:end]) > 0:
            y_pred_pa[start:end] = 1

    return y_pred_pa


def calculate_pa_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_pa = adjust_predictions(y_true, y_pred)
    return f1_score(y_true, y_pred_pa, zero_division=0)


def per_fault_analysis(
    y_test: np.ndarray,
    y_pred_raw: np.ndarray,
    run_test: List[int],
    sample_test: List[int],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fault_start_sample: int = 160
) -> pd.DataFrame:
    run_to_fault_map = pd.concat([
        val_df[["simulationRun", "faultNumber"]],
        test_df[["simulationRun", "faultNumber"]]
    ]).drop_duplicates().set_index("simulationRun")["faultNumber"].to_dict()

    df_results = pd.DataFrame({
        "run": run_test,
        "sample": sample_test,
        "faultNumber": [run_to_fault_map.get(r, 0) for r in run_test],
        "y_true": y_test,
        "y_pred": y_pred_raw
    })

    fault_analysis_results = []

    for fault_num in range(1, 21):
        df_fault = df_results[df_results["faultNumber"] == fault_num]
        if df_fault.empty:
            continue

        y_true_f = df_fault["y_true"].values
        y_pred_f = df_fault["y_pred"].values

        tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f, labels=[0, 1]).ravel()
        fdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        f1_std = f1_score(y_true_f, y_pred_f, zero_division=0)
        pa_f1 = calculate_pa_f1(y_true_f, y_pred_f)

        detection_delays = []
        fault_runs = df_fault["run"].unique()

        for run_id in fault_runs:
            df_run = df_fault[df_fault["run"] == run_id]
            first_detection = df_run[
                (df_run["y_pred"] == 1) &
                (df_run["sample"] >= fault_start_sample)
            ]
            if not first_detection.empty:
                delay = first_detection["sample"].min() - fault_start_sample
                detection_delays.append(delay)
            else:
                detection_delays.append(np.nan)

        mean_dd = np.nanmean(detection_delays)

        fault_analysis_results.append({
            "Fault Number": fault_num,
            "FDR": fdr,
            "FAR": far,
            "Mean DD": mean_dd,
            "F1 (Std)": f1_std,
            "PA_F1": pa_f1
        })

    df_analysis = pd.DataFrame(fault_analysis_results).set_index("Fault Number")
    df_avg = df_analysis.mean(axis=0, numeric_only=True).to_frame().T
    df_avg.index = ["Average"]

    return pd.concat([df_analysis, df_avg])


# =========================
# 主流程
# =========================
def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"找不到 threshold 檔案: {THRESHOLD_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"找不到 scaler 檔案: {SCALER_PATH}")
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"找不到 feature_cols 檔案: {FEATURE_COLS_PATH}")

    normal_path = DATA_DIR / "fault_free_training.csv"
    faulty_path = DATA_DIR / "faulty_testing_fault1to20.csv"

    # 載入 artifacts
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)
    best_threshold = float(threshold_info["threshold"])

    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print(f"[INFO] 使用 threshold: {best_threshold:.6f}")

    # 重建資料流程
    train_df, val_df, test_df = load_and_split_data(
        normal_path=normal_path,
        faulty_path=faulty_path,
        t=T,
        fault_start=FAULT_START
    )

    train_df, val_df, test_df = remove_dropped_features(
        train_df, val_df, test_df, feature_cols
    )

    train_df, val_df, test_df = apply_saved_scaler(
        train_df, val_df, test_df, feature_cols, scaler
    )

    X_train, y_train, run_train, sample_train = sliding_window(
        train_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )
    X_val, y_val, run_val, sample_val = sliding_window(
        val_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )
    X_test, y_test, run_test, sample_test = sliding_window(
        test_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )

    # 推論
    print("\n[INFO] 計算 train reconstruction error...")
    train_mae_loss = compute_reconstruction_mae(model, X_train)

    print("\n[INFO] 計算 val reconstruction error...")
    val_mae_loss = compute_reconstruction_mae(model, X_val)

    print("\n[INFO] 計算 test reconstruction error...")
    start_infer = time.time()
    test_mae_loss = compute_reconstruction_mae(model, X_test)
    infer_time = time.time() - start_infer
    print(f"[INFO] test inference time: {infer_time:.2f} sec")

    y_pred_raw = (test_mae_loss > best_threshold).astype(int)

    # 總體指標
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_raw, labels=[0, 1]).ravel()
    total_fdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    total_far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1_std = f1_score(y_test, y_pred_raw, zero_division=0)

    y_pred_pa = adjust_predictions(y_test, y_pred_raw)
    pa_f1_global = f1_score(y_test, y_pred_pa, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred_raw)

    fpr, tpr, _ = roc_curve(y_test, test_mae_loss)
    roc_auc = auc(fpr, tpr)

    pr_precisions, pr_recalls, _ = precision_recall_curve(y_test, test_mae_loss)
    pr_auc = auc(pr_recalls, pr_precisions)

    print("\n===== Overall Metrics =====")
    print(f"FDR: {total_fdr:.4f}")
    print(f"FAR: {total_far:.4f}")
    print(f"F1 : {f1_std:.4f}")
    print(f"PA_F1: {pa_f1_global:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC : {pr_auc:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred_raw, target_names=["Normal (0)", "Anomaly (1)"]))

    # Per-fault
    df_analysis = per_fault_analysis(
        y_test=y_test,
        y_pred_raw=y_pred_raw,
        run_test=run_test,
        sample_test=sample_test,
        val_df=val_df,
        test_df=test_df,
        fault_start_sample=FAULT_START
    )

    print("\n===== Per-Fault Analysis =====")
    print(df_analysis)

    df_analysis.to_csv(MODEL_DIR / "per_fault_analysis.csv")

    # 儲存總體 metrics
    overall_metrics = {
        "threshold": best_threshold,
        "FDR": float(total_fdr),
        "FAR": float(total_far),
        "F1": float(f1_std),
        "PA_F1": float(pa_f1_global),
        "MCC": float(mcc),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "test_inference_time_sec": float(infer_time),
        "avg_inference_time_ms_per_sample": float((infer_time / len(X_test)) * 1000.0),
    }
    with open(MODEL_DIR / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

    # 簡單畫 ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "roc_curve.png", dpi=150)
    plt.close()

    print(f"\n[INFO] 已輸出 {MODEL_DIR / 'per_fault_analysis.csv'}")
    print(f"[INFO] 已輸出 {MODEL_DIR / 'overall_metrics.json'}")
    print(f"[INFO] 已輸出 {MODEL_DIR / 'roc_curve.png'}")


if __name__ == "__main__":
    main()