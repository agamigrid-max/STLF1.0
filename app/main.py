import math
import os
import shutil
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Pipeline modules
from ingestion.load_data import load_csv
from preprocessing.clean_data import clean
from preprocessing.resample import enforce_hourly_frequency
from preprocessing.outliers import clip_outliers
from features.lags import add_lag_features
from features.rolling import add_rolling_features
from features.calendar import add_calendar_features
from models.baseline import train_baseline_model, predict_baseline

# Evaluation engine
from evaluation import evaluate_metrics


# ---------------------------------------------------
# JSON SANITIZER (fixes NaN/inf → None)
# ---------------------------------------------------
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# File paths
INPUT_PATH = "data/input/upload.csv"
OUTPUT_PATH = "data/output/forecast_output.csv"


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main UI page."""
    with open("app/templates/index.html") as f:
        return f.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file to data/input."""
    os.makedirs("data/input", exist_ok=True)
    with open(INPUT_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully"}


@app.post("/run")
def run_pipeline():
    """Run the full STLF pipeline with evaluation metrics."""

    print("\n================ PIPELINE START ================")

    # ---------------------------------------------------
    # 1. Load
    # ---------------------------------------------------
    df = load_csv(INPUT_PATH)
    print("STEP 1: Loaded DF shape:", df.shape)

    # ---------------------------------------------------
    # 2. Preprocess
    # ---------------------------------------------------
    df_clean = clean(df)
    df_clean = enforce_hourly_frequency(df_clean)
    df_clean = clip_outliers(df_clean)

    # ---------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------
    df_feat = add_lag_features(df_clean)
    df_feat = add_rolling_features(df_feat)
    df_feat = add_calendar_features(df_feat)

    # Drop NaNs from lag/rolling windows
    df_feat = df_feat.dropna()

    # ---------------------------------------------------
    # 4. Train baseline model
    # ---------------------------------------------------
    model, feature_cols = train_baseline_model(df_feat)

    # ---------------------------------------------------
    # 5. Predict
    # ---------------------------------------------------
    preds = predict_baseline(model, df_feat, feature_cols)

    # ---------------------------------------------------
    # 6. Save output CSV
    # ---------------------------------------------------
    output_df = df_feat.copy()
    output_df["forecast"] = preds

    os.makedirs("data/output", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # ---------------------------------------------------
    # 7. Compute evaluation metrics (v1)
    # ---------------------------------------------------
    y_true = df_feat["load"].values if "load" in df_feat.columns else None

    if y_true is not None:
        metrics = evaluate_metrics(
            y_true=np.array(y_true),
            y_pred=np.array(preds),
            horizons=list(range(1, len(preds) + 1)),
            runtime_stats={
                "training_time_sec": float("nan"),   # baseline model has no timing yet
                "inference_time_sec": float("nan"),
                "model_size_mb": float("nan"),
            },
            baseline_mape=None,
        )
    else:
        metrics = {"message": "No y_true column found for evaluation."}

    print("================ PIPELINE END ================\n")

    # ---------------------------------------------------
    # 8. Return JSON-safe response
    # ---------------------------------------------------
    return {
        "message": "Pipeline completed",
        "download_url": "/download",
        "metrics": sanitize_for_json(metrics),
    }


@app.get("/download")
def download_output():
    """Download the forecast output file."""
    return FileResponse(OUTPUT_PATH, filename="forecast_output.csv")
