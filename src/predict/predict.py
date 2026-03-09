import joblib
import pandas as pd 

def model_predict(X ,path):
    model = joblib.load(path)
    return model.predict(X)

def save_predictions(y_pred, path):
    y_pred = pd.DataFrame({"prediction": y_pred})
    y_pred.to_csv(path, index=False)