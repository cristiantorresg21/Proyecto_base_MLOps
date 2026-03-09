from pathlib import Path
cwd = Path.cwd()

## Directorios 
BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TEST_DATASET_DIR = DATA_DIR / "tests"
TEST_DATASET_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

## Archivos
DATASET_PATH = DATA_DIR / "house_data_washington.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# 


