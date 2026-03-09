import argparse
import joblib
import pandas as pd 

from configs.paths import TEST_DATASET_DIR, SCALER_PATH, MODEL_PATH, REPORTS_DIR
from configs.configs import FEATURES
from src.data.load_data import load_data
from src.preprocess.preprocess import transform_scaler
from src.predict.predict import model_predict, save_predictions

parser = argparse.ArgumentParser(description="Inferencia de un modelo de regresion Ridge")
parser.add_argument("--dataset", "-d", default="dataset.csv", type=str, help="Dataset ocupado para realizar la inferencia")
args = parser.parse_args()
DATASET = args.dataset
PRED_DATASET = f"pred_{DATASET}"
TEST_DATASET = TEST_DATASET_DIR / DATASET
PRED_DATASET_TEST_PATH = REPORTS_DIR / PRED_DATASET

def main():
    X, _ = load_data(TEST_DATASET, FEATURES)

    scaler = joblib.load(SCALER_PATH)

    X = transform_scaler(X, scaler=scaler)

    y_pred = model_predict(X, MODEL_PATH)

    print("Predicciones generadas:", len(y_pred))

    save_predictions(y_pred, PRED_DATASET_TEST_PATH)

    print(f"Predicciones guardadas en: {PRED_DATASET_TEST_PATH}")

if __name__ == "__main__":
    main()
