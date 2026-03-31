from pathlib import Path
import json
from typing import List, Optional

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


app = FastAPI(
    title="LSTM-AE Anomaly Detection API",
    version="1.0.0",
    description="Inference API for TEP-based LSTM Autoencoder anomaly detection."
)

model = None
scaler = None
threshold = None
feature_cols = None
metadata = None


# =========================
# Request / Response Models
# =========================
class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Single window input with shape [timesteps, features]."
    )


class PredictBatchRequest(BaseModel):
    sequences: List[List[List[float]]] = Field(
        ...,
        description="Batch window input with shape [batch, timesteps, features]."
    )


class PredictFullRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Full raw sequence with shape [total_timesteps, features]. API will auto-window it."
    )
    aggregate: str = Field(
        default="max",
        description="Aggregation method across windows: max, mean, proportion"
    )


class PredictFullBatchRequest(BaseModel):
    sequences: List[List[List[float]]] = Field(
        ...,
        description="Batch raw sequences. Each element is [total_timesteps, features]."
    )
    aggregate: str = Field(
        default="max",
        description="Aggregation method across windows: max, mean, proportion"
    )
    
class PredictTEPRecordsRequest(BaseModel):
    records: List[dict] = Field(
        ...,
        description="Raw TEP records. API will automatically keep only trained feature columns."
    )
    aggregate: str = Field(
        default="max",
        description="Aggregation method across windows: max, mean, proportion"
    )

class PredictTEPOfficialTestRequest(BaseModel):
    records: List[dict] = Field(
        ...,
        description="Raw records from official TEP test dataset."
    )
    run_length: int = Field(
        default=960,
        description="Number of timesteps per simulation run."
    )
    fault_start: int = Field(
        default=160,
        description="Fault onset sample index. sample < fault_start is treated as normal."
    )
    aggregate: str = Field(
        default="max",
        description="Aggregation method across windows: max, mean, proportion"
    )
# =========================
# Artifact Loading
# =========================
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Missing threshold file: {THRESHOLD_PATH}")
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"Missing feature columns file: {FEATURE_COLS_PATH}")

    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)
    loaded_threshold = float(threshold_info["threshold"])

    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        loaded_feature_cols = json.load(f)

    loaded_metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            loaded_metadata = json.load(f)

    return loaded_model, loaded_scaler, loaded_threshold, loaded_feature_cols, loaded_metadata


@app.on_event("startup")
def startup_event():
    global model, scaler, threshold, feature_cols, metadata
    model, scaler, threshold, feature_cols, metadata = load_artifacts()


# =========================
# Utility Functions
# =========================
def get_window_size() -> int:
    return int(metadata.get("window_size", 20))


def get_expected_features() -> int:
    return len(feature_cols)


def validate_2d_sequence(x: np.ndarray, name: str = "sequence") -> None:
    if x.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be 2D with shape [timesteps, features]"
        )

    expected_features = get_expected_features()
    if x.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature dimension mismatch for {name}. "
                f"Expected {expected_features}, got {x.shape[1]}"
            )
        )


def scale_sequence(x: np.ndarray) -> np.ndarray:
    return scaler.transform(x)


def compute_window_score(x_window_scaled: np.ndarray) -> float:
    x_input = np.expand_dims(x_window_scaled, axis=0)  # [1, T, F]
    x_pred = model.predict(x_input, verbose=0)
    score = float(np.mean(np.abs(x_pred - x_input), axis=(1, 2))[0])
    return score


def make_windows(x_scaled: np.ndarray, window_size: int) -> np.ndarray:
    total_len = x_scaled.shape[0]
    if total_len < window_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence length is too short for windowing. "
                f"Need at least {window_size} timesteps, got {total_len}"
            )
        )

    windows = []
    for i in range(total_len - window_size + 1):
        windows.append(x_scaled[i:i + window_size])

    return np.array(windows, dtype=np.float32)


def compute_window_scores_batch(windows: np.ndarray) -> np.ndarray:
    preds = model.predict(windows, verbose=0)
    scores = np.mean(np.abs(preds - windows), axis=(1, 2))
    return scores


def aggregate_scores(scores: np.ndarray, method: str = "max") -> float:
    method = method.lower()

    if len(scores) == 0:
        raise HTTPException(status_code=400, detail="No windows generated for aggregation.")

    if method == "max":
        return float(np.max(scores))
    if method == "mean":
        return float(np.mean(scores))
    if method == "proportion":
        # 回傳超過 threshold 的比例
        return float(np.mean(scores > threshold))

    raise HTTPException(
        status_code=400,
        detail="Invalid aggregate method. Use one of: max, mean, proportion"
    )


def build_window_result(score: float) -> dict:
    return {
        "anomaly_score": float(score),
        "threshold": float(threshold),
        "is_anomaly": bool(score > threshold)
    }

def records_to_feature_matrix(records: List[dict], feature_cols: List[str]) -> np.ndarray:
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    # 如果都有 sample，就依 sample 排序
    if all("sample" in r for r in records):
        try:
            records = sorted(records, key=lambda r: float(r["sample"]))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="sample column exists but cannot be used for sorting."
            )

    # 檢查必要欄位是否缺少
    missing_cols = set()
    for record in records:
        for col in feature_cols:
            if col not in record:
                missing_cols.add(col)

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Some required feature columns are missing.",
                "missing_columns_example": sorted(list(missing_cols))[:10],
                "expected_feature_count": len(feature_cols)
            }
        )

    # 只保留 feature_cols，其他欄位自動忽略
    matrix = []
    for i, record in enumerate(records):
        row = []
        for col in feature_cols:
            value = record[col]

            if value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Null value found at record {i}, column '{col}'."
                )

            try:
                row.append(float(value))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Non-numeric value found at record {i}, column '{col}': {value}"
                )
        matrix.append(row)

    return np.array(matrix, dtype=np.float32)
    
def preprocess_official_tep_test_records(
    records: List[dict],
    run_length: int,
    fault_start: int
) -> List[dict]:
    """
    For official TEP test data:
    - rebuild simulationRun and sample if needed
    - create label using fault_start rule
    """
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    processed = []

    for idx, record in enumerate(records):
        r = dict(record)

        # 若沒有 simulationRun / sample，依官方格式重建
        if "simulationRun" not in r:
            r["simulationRun"] = idx // run_length

        if "sample" not in r:
            r["sample"] = idx % run_length

        # faultNumber 若不存在，預設 0
        fault_number = int(r.get("faultNumber", 0))

        # 建 label（只用於分析，不進模型）
        r["label"] = int((fault_number > 0) and (int(r["sample"]) >= fault_start))

        processed.append(r)

    return processed
# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def get_metadata():
    return {
        "expected_features": get_expected_features(),
        "window_size": get_window_size(),
        "threshold": float(threshold),
        "model_path": MODEL_PATH.name,
        "available_endpoints": [
            "/health",
            "/metadata",
            "/predict",
            "/predict_batch",
            "/predict_full",
            "/predict_full_batch",
        ]
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        x = np.array(req.sequence, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    validate_2d_sequence(x, name="sequence")

    expected_window = get_window_size()
    if x.shape[0] != expected_window:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Window length mismatch. Expected {expected_window} timesteps, "
                f"got {x.shape[0]}"
            )
        )

    x_scaled = scale_sequence(x)
    score = compute_window_score(x_scaled)

    return build_window_result(score)


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if len(req.sequences) == 0:
        raise HTTPException(status_code=400, detail="sequences cannot be empty.")

    expected_window = get_window_size()
    results = []

    for idx, seq in enumerate(req.sequences):
        try:
            x = np.array(seq, dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input at index {idx}: {str(e)}")

        validate_2d_sequence(x, name=f"sequences[{idx}]")

        if x.shape[0] != expected_window:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Window length mismatch at index {idx}. "
                    f"Expected {expected_window}, got {x.shape[0]}"
                )
            )

        x_scaled = scale_sequence(x)
        score = compute_window_score(x_scaled)
        results.append({
            "index": idx,
            "anomaly_score": float(score),
            "threshold": float(threshold),
            "is_anomaly": bool(score > threshold)
        })

    return {
        "count": len(results),
        "results": results
    }


@app.post("/predict_full")
def predict_full(req: PredictFullRequest):
    try:
        x = np.array(req.sequence, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    validate_2d_sequence(x, name="sequence")

    x_scaled = scale_sequence(x)
    window_size = get_window_size()
    windows = make_windows(x_scaled, window_size=window_size)
    scores = compute_window_scores_batch(windows)

    aggregate_score = aggregate_scores(scores, method=req.aggregate)

    if req.aggregate.lower() == "proportion":
        is_anomaly = aggregate_score > 0.0
    else:
        is_anomaly = aggregate_score > threshold

    return {
        "aggregate_method": req.aggregate,
        "num_windows": int(len(scores)),
        "window_size": int(window_size),
        "max_window_score": float(np.max(scores)),
        "mean_window_score": float(np.mean(scores)),
        "aggregate_score": float(aggregate_score),
        "threshold": float(threshold),
        "is_anomaly": bool(is_anomaly),
        "window_predictions": (scores > threshold).astype(int).tolist(),
        "window_scores": scores.tolist()
    }


@app.post("/predict_full_batch")
def predict_full_batch(req: PredictFullBatchRequest):
    if len(req.sequences) == 0:
        raise HTTPException(status_code=400, detail="sequences cannot be empty.")

    results = []
    window_size = get_window_size()

    for idx, seq in enumerate(req.sequences):
        try:
            x = np.array(seq, dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input at index {idx}: {str(e)}")

        validate_2d_sequence(x, name=f"sequences[{idx}]")

        x_scaled = scale_sequence(x)
        windows = make_windows(x_scaled, window_size=window_size)
        scores = compute_window_scores_batch(windows)

        aggregate_score = aggregate_scores(scores, method=req.aggregate)

        if req.aggregate.lower() == "proportion":
            is_anomaly = aggregate_score > 0.0
        else:
            is_anomaly = aggregate_score > threshold

        results.append({
            "index": idx,
            "aggregate_method": req.aggregate,
            "num_windows": int(len(scores)),
            "max_window_score": float(np.max(scores)),
            "mean_window_score": float(np.mean(scores)),
            "aggregate_score": float(aggregate_score),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly)
        })

    return {
        "count": len(results),
        "window_size": int(window_size),
        "results": results
    }
    
@app.post("/predict_tep_records")
def predict_tep_records(req: PredictTEPRecordsRequest):
    x = records_to_feature_matrix(req.records, feature_cols)

    # 檢查 2D shape
    if x.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="records must form a 2D matrix [timesteps, features]"
        )

    expected_features = len(feature_cols)
    if x.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Feature dimension mismatch. Expected {expected_features}, got {x.shape[1]}"
        )

    # scaler
    x_scaled = scaler.transform(x)

    # windowing
    window_size = metadata.get("window_size", 20)
    if x_scaled.shape[0] < window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {window_size} records, got {x_scaled.shape[0]}"
        )

    windows = []
    for i in range(x_scaled.shape[0] - window_size + 1):
        windows.append(x_scaled[i:i + window_size])

    windows = np.array(windows, dtype=np.float32)

    # prediction
    preds = model.predict(windows, verbose=0)
    scores = np.mean(np.abs(preds - windows), axis=(1, 2))

    # aggregate
    agg = req.aggregate.lower()
    if agg == "max":
        aggregate_score = float(np.max(scores))
        is_anomaly = aggregate_score > threshold
    elif agg == "mean":
        aggregate_score = float(np.mean(scores))
        is_anomaly = aggregate_score > threshold
    elif agg == "proportion":
        aggregate_score = float(np.mean(scores > threshold))
        is_anomaly = aggregate_score > 0.0
    else:
        raise HTTPException(
            status_code=400,
            detail="aggregate must be one of: max, mean, proportion"
        )

    return {
        "input_mode": "raw_tep_records",
        "aggregate_method": agg,
        "num_records": int(x.shape[0]),
        "num_features_used": int(x.shape[1]),
        "window_size": int(window_size),
        "num_windows": int(len(scores)),
        "max_window_score": float(np.max(scores)),
        "mean_window_score": float(np.mean(scores)),
        "aggregate_score": float(aggregate_score),
        "threshold": float(threshold),
        "is_anomaly": bool(is_anomaly),
        "window_predictions": (scores > threshold).astype(int).tolist(),
        "window_scores": scores.tolist(),
        "used_feature_columns": feature_cols,
        "ignored_columns": sorted(
            list(set(req.records[0].keys()) - set(feature_cols))
        ) if len(req.records) > 0 else []
    }

@app.post("/predict_tep_official_test_records")
def predict_tep_official_test_records(req: PredictTEPOfficialTestRequest):
    # 1. 先依 TEP 官方規則補 simulationRun / sample / label
    processed_records = preprocess_official_tep_test_records(
        records=req.records,
        run_length=req.run_length,
        fault_start=req.fault_start
    )

    # 2. 轉成模型特徵矩陣（自動忽略 faultNumber / sample / label）
    x = records_to_feature_matrix(processed_records, feature_cols)

    if x.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="records must form a 2D matrix [timesteps, features]"
        )

    expected_features = len(feature_cols)
    if x.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Feature dimension mismatch. Expected {expected_features}, got {x.shape[1]}"
        )

    # 3. scaler
    x_scaled = scale_sequence(x)

    # 4. sliding window
    window_size = get_window_size()
    windows = make_windows(x_scaled, window_size=window_size)

    # 5. predict
    scores = compute_window_scores_batch(windows)

    agg = req.aggregate.lower()
    aggregate_score = aggregate_scores(scores, method=agg)

    if agg == "proportion":
        is_anomaly = aggregate_score > 0.0
    else:
        is_anomaly = aggregate_score > threshold

    # 6. 同步建立 window-level label（方便分析）
    labels = np.array([r["label"] for r in processed_records], dtype=int)
    window_labels = []
    for i in range(len(labels) - window_size + 1):
        window_labels.append(int(labels[i:i + window_size].max()))

    return {
        "input_mode": "official_tep_test_records",
        "aggregate_method": agg,
        "run_length": int(req.run_length),
        "fault_start": int(req.fault_start),
        "num_records": int(len(processed_records)),
        "num_features_used": int(x.shape[1]),
        "window_size": int(window_size),
        "num_windows": int(len(scores)),
        "max_window_score": float(np.max(scores)),
        "mean_window_score": float(np.mean(scores)),
        "aggregate_score": float(aggregate_score),
        "threshold": float(threshold),
        "is_anomaly": bool(is_anomaly),
        "window_predictions": (scores > threshold).astype(int).tolist(),
        "window_labels": window_labels,
        "window_scores": scores.tolist(),
        "used_feature_columns": feature_cols,
        "ignored_columns": sorted(
            list(set(processed_records[0].keys()) - set(feature_cols))
        ) if len(processed_records) > 0 else []
    }