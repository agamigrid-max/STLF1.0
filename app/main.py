from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

# Import pipeline modules
from ingestion.load_data import load_csv
from preprocessing.clean_data import clean
from preprocessing.resample import enforce_hourly_frequency
from preprocessing.outliers import clip_outliers
from features.lags import add_lag_features
from features.rolling import add_rolling_features
from features.calendar import add_calendar_features
from models.baseline import train_baseline_model, predict_baseline

app = FastAPI()

# Serve static files (CSS)
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
    """Run the full STLF pipeline with debug logging."""

    print("\n================ PIPELINE START ================")

    # Load
    df = load_csv(INPUT_PATH)
    print("STEP 1: Loaded DF shape:", df.shape)
    print(df.head())

    # Preprocess
    df_clean = clean(df)
    print("STEP 2: After clean:", df_clean.shape)

    df_clean = enforce_hourly_frequency(df_clean)
    print("STEP 3: After resample:", df_clean.shape)

    df_clean = clip_outliers(df_clean)
    print("STEP 4: After outlier clipping:", df_clean.shape)

    # Feature engineering
    df_feat = add_lag_features(df_clean)
    print("STEP 5: After lag features:", df_feat.shape)

    print("DEBUG: entering rolling features")
    df_feat = add_rolling_features(df_feat)
    print("STEP 6: After rolling features:", df_feat.shape)

    print("DEBUG: entering calendar features")
    df_feat = add_calendar_features(df_feat)
    print("STEP 7: After calendar features:", df_feat.shape)
    
    # FIX: drop NaNs created by lag + rolling windows
    df_feat = df_feat.dropna()
    print("STEP 7.5: After dropna:", df_feat.shape)


    print("DEBUG: entering model training")
    model, feature_cols = train_baseline_model(df_feat)
    print("STEP 8: Model trained. Feature cols:", feature_cols)

    print("DEBUG: entering prediction")
    preds = predict_baseline(model, df_feat, feature_cols)
    print("STEP 9: Predictions generated. Length:", len(preds))

    # Build output DataFrame
    output_df = df_feat.copy()
    output_df["forecast"] = preds

    # Save output
    os.makedirs("data/output", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("STEP 10: Output saved to:", OUTPUT_PATH)
    print("================ PIPELINE END ================\n")

    return {"message": "Pipeline completed", "download_url": "/download"}


@app.get("/download")
def download_output():
    """Download the forecast output file."""
    return FileResponse(OUTPUT_PATH, filename="forecast_output.csv")
