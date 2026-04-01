from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    classification_report,
)


# =========================
# 路徑設定
# =========================
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
EVAL_OUTPUT_DIR = ROOT / "evaluation_outputs"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="LSTM-AE Anomaly Detection API",
    version="3.0.0",
    description="Simplified inference, evaluation, and dashboard API for official TEP test records."
)

app.mount("/evaluation_outputs", StaticFiles(directory=EVAL_OUTPUT_DIR), name="evaluation_outputs")

model = None
scaler = None
threshold = None
feature_cols = None
metadata = None
latest_evaluation_result = None


# =========================
# Request Models
# =========================
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


class EvaluateTEPOfficialTestRequest(BaseModel):
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
    save_plots: bool = Field(
        default=True,
        description="Whether to save evaluation plots locally."
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


def scale_sequence(x: np.ndarray) -> np.ndarray:
    return scaler.transform(x)


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
        return float(np.mean(scores > threshold))

    raise HTTPException(
        status_code=400,
        detail="Invalid aggregate method. Use one of: max, mean, proportion"
    )


def records_to_feature_matrix(records: List[dict], feature_cols: List[str]) -> np.ndarray:
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    if all("sample" in r for r in records):
        try:
            records = sorted(records, key=lambda r: float(r["sample"]))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="sample column exists but cannot be used for sorting."
            )

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
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    processed = []

    for idx, record in enumerate(records):
        r = dict(record)

        if "simulationRun" not in r:
            r["simulationRun"] = idx // run_length

        if "sample" not in r:
            r["sample"] = idx % run_length

        fault_number = int(r.get("faultNumber", 0))
        r["label"] = int((fault_number > 0) and (int(r["sample"]) >= fault_start))

        processed.append(r)

    return processed


def run_official_test_inference(
    processed_records: List[dict],
    aggregate: str
) -> dict:
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

    x_scaled = scale_sequence(x)
    window_size = get_window_size()
    windows = make_windows(x_scaled, window_size=window_size)
    scores = compute_window_scores_batch(windows)

    agg = aggregate.lower()
    aggregate_score = aggregate_scores(scores, method=agg)

    if agg == "proportion":
        is_anomaly = aggregate_score > 0.0
    else:
        is_anomaly = aggregate_score > threshold

    labels = np.array([r["label"] for r in processed_records], dtype=int)
    window_labels = []
    for i in range(len(labels) - window_size + 1):
        window_labels.append(int(labels[i:i + window_size].max()))

    return {
        "x": x,
        "scores": scores,
        "window_labels": window_labels,
        "window_predictions": (scores > threshold).astype(int).tolist(),
        "aggregate_score": float(aggregate_score),
        "is_anomaly": bool(is_anomaly),
        "window_size": int(window_size),
        "num_windows": int(len(scores)),
        "max_window_score": float(np.max(scores)),
        "mean_window_score": float(np.mean(scores)),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(pr_recall, pr_precision)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "fdr": float(fdr),
        "far": float(far),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "num_windows": int(len(y_true)),
        "threshold": float(threshold),
    }


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metrics: Dict,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EVAL_OUTPUT_DIR / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(pr_recall, pr_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recall, pr_precision, label=f"AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal (0)", "Anomaly (1)"],
        zero_division=0
    )
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return str(output_dir)


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata_api():
    return {
        "expected_features": get_expected_features(),
        "window_size": get_window_size(),
        "threshold": float(threshold),
        "model_path": MODEL_PATH.name,
        "available_endpoints": [
            "/health",
            "/metadata",
            "/predict_tep_official_test_records",
            "/evaluate_tep_official_test_records",
            "/latest_evaluation",
        ]
    }


@app.post("/predict_tep_official_test_records")
def predict_tep_official_test_records(req: PredictTEPOfficialTestRequest):
    processed_records = preprocess_official_tep_test_records(
        records=req.records,
        run_length=req.run_length,
        fault_start=req.fault_start
    )

    result = run_official_test_inference(
        processed_records=processed_records,
        aggregate=req.aggregate
    )

    return {
        "input_mode": "official_tep_test_records",
        "aggregate_method": req.aggregate.lower(),
        "run_length": int(req.run_length),
        "fault_start": int(req.fault_start),
        "num_records": int(len(processed_records)),
        "num_features_used": int(result["x"].shape[1]),
        "window_size": int(result["window_size"]),
        "num_windows": int(result["num_windows"]),
        "max_window_score": float(result["max_window_score"]),
        "mean_window_score": float(result["mean_window_score"]),
        "aggregate_score": float(result["aggregate_score"]),
        "threshold": float(threshold),
        "is_anomaly": bool(result["is_anomaly"]),
        "window_predictions": result["window_predictions"],
        "window_labels": result["window_labels"],
        "window_scores": result["scores"].tolist(),
        "used_feature_columns": feature_cols,
        "ignored_columns": sorted(
            list(set(processed_records[0].keys()) - set(feature_cols))
        ) if len(processed_records) > 0 else []
    }


@app.post("/evaluate_tep_official_test_records")
def evaluate_tep_official_test_records(req: EvaluateTEPOfficialTestRequest):
    global latest_evaluation_result

    processed_records = preprocess_official_tep_test_records(
        records=req.records,
        run_length=req.run_length,
        fault_start=req.fault_start
    )

    result = run_official_test_inference(
        processed_records=processed_records,
        aggregate=req.aggregate
    )

    y_true = np.array(result["window_labels"], dtype=int)
    y_pred = np.array(result["window_predictions"], dtype=int)
    y_score = np.array(result["scores"], dtype=float)

    metrics = evaluate_predictions(y_true, y_pred, y_score)

    output_dir = None
    plot_urls = None

    if req.save_plots:
        output_dir = save_evaluation_artifacts(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics
        )

        eval_name = Path(output_dir).name
        plot_urls = {
            "confusion_matrix": f"/evaluation_outputs/{eval_name}/confusion_matrix.png",
            "roc_curve": f"/evaluation_outputs/{eval_name}/roc_curve.png",
            "pr_curve": f"/evaluation_outputs/{eval_name}/pr_curve.png",
        }

    latest_evaluation_result = {
        "input_mode": "official_tep_test_records_evaluation",
        "aggregate_method": req.aggregate.lower(),
        "run_length": int(req.run_length),
        "fault_start": int(req.fault_start),
        "num_records": int(len(processed_records)),
        "window_size": int(result["window_size"]),
        "num_windows": int(result["num_windows"]),
        "metrics": metrics,
        "output_dir": output_dir,
        "plot_urls": plot_urls,
        "window_predictions": result["window_predictions"],
        "window_labels": result["window_labels"],
        "window_scores": result["scores"].tolist(),
        "message": "evaluation completed"
    }

    return latest_evaluation_result


@app.get("/latest_evaluation", response_class=HTMLResponse)
def latest_evaluation():
    if latest_evaluation_result is None:
        return """
        <html>
        <head><title>Latest Evaluation</title></head>
        <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h1>Latest Evaluation</h1>
            <p>No evaluation result yet.</p>
            <p>Please call <code>/evaluate_tep_official_test_records</code> first.</p>
        </body>
        </html>
        """

    metrics = latest_evaluation_result["metrics"]
    plot_urls = latest_evaluation_result.get("plot_urls") or {}

    confusion_matrix_url = plot_urls.get("confusion_matrix", "")
    roc_curve_url = plot_urls.get("roc_curve", "")
    pr_curve_url = plot_urls.get("pr_curve", "")

    html = f"""
    <html>
    <head>
        <title>Latest Evaluation Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 24px;
                background: #f7f9fc;
                color: #222;
            }}
            h1 {{
                margin-bottom: 8px;
            }}
            .sub {{
                color: #666;
                margin-bottom: 24px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 32px;
            }}
            .card {{
                background: white;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .metric-name {{
                font-size: 14px;
                color: #666;
                margin-bottom: 8px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
            }}
            .section {{
                margin-top: 24px;
            }}
            .image-card {{
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}
            .small {{
                color: #777;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <h1>Latest Evaluation Dashboard</h1>
        <div class="sub">
            TEP Official Test Evaluation Result
        </div>

        <div class="grid">
            <div class="card">
                <div class="metric-name">Accuracy</div>
                <div class="metric-value">{metrics['accuracy']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">Precision</div>
                <div class="metric-value">{metrics['precision']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">Recall</div>
                <div class="metric-value">{metrics['recall']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">F1</div>
                <div class="metric-value">{metrics['f1']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">MCC</div>
                <div class="metric-value">{metrics['mcc']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">FDR</div>
                <div class="metric-value">{metrics['fdr']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">FAR</div>
                <div class="metric-value">{metrics['far']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">ROC AUC</div>
                <div class="metric-value">{metrics['roc_auc']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">PR AUC</div>
                <div class="metric-value">{metrics['pr_auc']:.4f}</div>
            </div>
            <div class="card">
                <div class="metric-name">Num Windows</div>
                <div class="metric-value">{metrics['num_windows']}</div>
            </div>
            <div class="card">
                <div class="metric-name">Threshold</div>
                <div class="metric-value">{metrics['threshold']:.4f}</div>
            </div>
        </div>

        <div class="section">
            <div class="image-card">
                <h2>Confusion Matrix</h2>
                <img src="{confusion_matrix_url}" alt="Confusion Matrix">
            </div>

            <div class="image-card">
                <h2>ROC Curve</h2>
                <img src="{roc_curve_url}" alt="ROC Curve">
            </div>

            <div class="image-card">
                <h2>PR Curve</h2>
                <img src="{pr_curve_url}" alt="PR Curve">
            </div>
        </div>

        <div class="small">
            output_dir: {latest_evaluation_result.get("output_dir")}
        </div>
    </body>
    </html>
    """
    return html
