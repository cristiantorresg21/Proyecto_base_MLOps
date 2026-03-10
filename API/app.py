from fastapi import FastAPI, Body
import pandas as pd
import joblib

from configs.paths import SCALER_PATH, MODEL_PATH
from configs.configs import FEATURES


scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI()

@app.post("/predict")
def predict(data: list = Body(...)):
    try:
        X = pd.DataFrame(data, columns=FEATURES)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}