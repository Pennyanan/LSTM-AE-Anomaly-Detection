from pathlib import Path
from typing import List, Tuple
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =========================
# 專案路徑設定
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


# =========================
# 參數設定
# =========================
T = 960
FAULT_START = 160
WINDOW_SIZE = 20
STRIDE = 1
SEED = 42

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"


# =========================
# 固定隨機種子
# =========================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# 1. 標註函式
# =========================
def set_label_based_on_sample(df: pd.DataFrame, onset: int = 160) -> pd.DataFrame:
    d = df.copy()
    d["label"] = ((d["faultNumber"] > 0) & (d["sample"] >= onset)).astype(int)
    return d


# =========================
# 2. 資料讀取與切分
# =========================
def load_and_split_data(
    normal_path: Path,
    faulty_path: Path,
    t: int = 960,
    fault_start: int = 160
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_path)
    df_faulty = pd.read_csv(faulty_path)

    # 正常資料
    n_run_normal = len(df_normal) // t
    df_normal = df_normal.iloc[:n_run_normal * t].copy()
    df_normal["simulationRun"] = np.repeat(np.arange(n_run_normal), t)
    df_normal["sample"] = np.tile(np.arange(t), n_run_normal)
    df_normal["faultNumber"] = 0
    df_normal["label"] = 0

    # 異常資料
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

    print(f"[INFO] train_df shape: {train_df.shape}")
    print(f"[INFO] val_df   shape: {val_df.shape}")
    print(f"[INFO] test_df  shape: {test_df.shape}")

    return train_df, val_df, test_df


# =========================
# 3. 特徵欄位
# =========================
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("xmeas_") or col.startswith("xmv_")]


# =========================
# 4. 移除低變異特徵
# =========================
def remove_low_variance_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 1e-6
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], pd.DataFrame]:
    feature_cols = get_feature_columns(train_df)
    stds = train_df[feature_cols].std().sort_values()

    std_df = stds.reset_index()
    std_df.columns = ["feature", "std"]

    features_to_drop = std_df.loc[std_df["std"] < threshold, "feature"].tolist()

    train_df = train_df.drop(columns=features_to_drop)
    val_df = val_df.drop(columns=features_to_drop)
    test_df = test_df.drop(columns=features_to_drop)

    final_feature_cols = get_feature_columns(train_df)

    print(f"[INFO] 已移除 {len(features_to_drop)} 個低變異特徵")
    return train_df, val_df, test_df, final_feature_cols, features_to_drop, std_df


# =========================
# 5. Z-score
# =========================
def apply_zscore(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, pd.DataFrame]:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    zscore_stats = pd.DataFrame({
        "feature": feature_cols,
        "mean": scaler.mean_,
        "std": scaler.scale_
    })

    return train_df, val_df, test_df, scaler, zscore_stats


# =========================
# 6. Sliding Window
# =========================
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
# 7. 建立 LSTM Autoencoder
# =========================
def build_lstm_autoencoder(timesteps: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, input_shape=(timesteps, n_features), return_sequences=False),
        Dropout(0.2),
        RepeatVector(timesteps),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(n_features))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae"
    )
    return model


# =========================
# 8. 訓練模型
# =========================
def train_lstm_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_path: Path
) -> Tuple[tf.keras.Model, dict, float]:
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_lstm_autoencoder(timesteps, n_features)
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        mode="min",
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\n[INFO] 開始訓練 LSTM Autoencoder...")
    start_time = time.time()

    history = model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        shuffle=False,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"[INFO] 訓練完成，耗時: {training_time:.2f} 秒")

    best_model = tf.keras.models.load_model(str(model_path))
    return best_model, history.history, training_time


# =========================
# 9. 重構誤差
# =========================
def compute_reconstruction_mae(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    X_pred = model.predict(X, verbose=1)
    mae_loss = np.mean(np.abs(X_pred - X), axis=(1, 2))
    return mae_loss


# =========================
# 10. 儲存前處理 artifacts
# =========================
def save_preprocessing_artifacts(
    scaler: StandardScaler,
    feature_cols: List[str],
    dropped_features: List[str],
    zscore_stats: pd.DataFrame,
    std_df: pd.DataFrame
) -> None:
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(MODEL_DIR / "dropped_features.json", "w", encoding="utf-8") as f:
        json.dump(dropped_features, f, ensure_ascii=False, indent=2)

    zscore_stats.to_csv(MODEL_DIR / "zscore_stats.csv", index=False)
    std_df.to_csv(MODEL_DIR / "feature_std_report.csv", index=False)


# =========================
# 11. 儲存 threshold
# =========================
def save_threshold(threshold: float, strategy: str = "train_99_percentile") -> None:
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "strategy": strategy
            },
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# 12. 主流程
# =========================
def main() -> None:
    set_seed(SEED)

    normal_path = DATA_DIR / "fault_free_training.csv"
    faulty_path = DATA_DIR / "faulty_testing_fault1to20.csv"

    if not normal_path.exists():
        raise FileNotFoundError(f"找不到正常資料檔案: {normal_path}")
    if not faulty_path.exists():
        raise FileNotFoundError(f"找不到異常資料檔案: {faulty_path}")

    # 1) 讀取與切分
    train_df, val_df, test_df = load_and_split_data(
        normal_path=normal_path,
        faulty_path=faulty_path,
        t=T,
        fault_start=FAULT_START
    )

    # 2) 移除低變異特徵
    train_df, val_df, test_df, feature_cols, dropped_features, std_df = remove_low_variance_features(
        train_df, val_df, test_df, threshold=1e-6
    )

    # 3) 標準化
    train_df, val_df, test_df, scaler, zscore_stats = apply_zscore(
        train_df, val_df, test_df, feature_cols
    )

    # 4) 滑動視窗
    X_train, y_train, run_train, sample_train = sliding_window(
        train_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )
    X_val, y_val, run_val, sample_val = sliding_window(
        val_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )
    X_test, y_test, run_test, sample_test = sliding_window(
        test_df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE
    )

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_val shape:   {X_val.shape}")
    print(f"[INFO] X_test shape:  {X_test.shape}")

    # 5) 儲存 preprocessing artifacts
    save_preprocessing_artifacts(
        scaler=scaler,
        feature_cols=feature_cols,
        dropped_features=dropped_features,
        zscore_stats=zscore_stats,
        std_df=std_df
    )

    # 6) 訓練模型
    model, history_dict, training_time = train_lstm_autoencoder(
        X_train=X_train,
        X_val=X_val,
        model_path=MODEL_PATH
    )

    # 7) 訓練集與驗證集重構誤差
    print("\n[INFO] 計算 train reconstruction error...")
    train_mae_loss = compute_reconstruction_mae(model, X_train)

    print("\n[INFO] 計算 val reconstruction error...")
    val_mae_loss = compute_reconstruction_mae(model, X_val)

    # 8) threshold
    robust_threshold = np.percentile(train_mae_loss, 99)
    best_threshold = float(robust_threshold)

    print(f"[INFO] threshold = {best_threshold:.6f}")

    # 9) 儲存 threshold 與誤差
    save_threshold(best_threshold, strategy="train_99_percentile")
    np.save(MODEL_DIR / "train_mae_loss.npy", train_mae_loss)
    np.save(MODEL_DIR / "val_mae_loss.npy", val_mae_loss)

    # 10) 也把 windowed test 存起來，後面 evaluate.py 會用
    np.save(MODEL_DIR / "X_test.npy", X_test)
    np.save(MODEL_DIR / "y_test.npy", y_test)

    # 11) 儲存 metadata
    metadata = {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "fault_start": FAULT_START,
        "seed": SEED,
        "training_time_sec": float(training_time),
        "n_train_windows": int(len(X_train)),
        "n_val_windows": int(len(X_val)),
        "n_test_windows": int(len(X_test)),
        "n_features": int(X_train.shape[2])
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[INFO] 訓練流程完成")
    print(f"[INFO] 模型: {MODEL_PATH}")
    print(f"[INFO] threshold: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()