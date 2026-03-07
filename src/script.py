import argparse
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib

from paths import DATASET_PATH, MODEL_PATH, SCALER_PATH, METRICS_PATH

parser = argparse.ArgumentParser(description='Modelo de regresión lineal con regularización Ridge')
parser.add_argument('--alpha', '-a', default=0, type=float, help='El parametro alpha del modelo Ridge')
parser.add_argument('--seed', '-s' , default=45, type=int, help='Semilla de los numeros aleatorios')
args = parser.parse_args()
ALPHA = args.alpha
RANDOM_SEED = args.seed

def load_data(path: str):
    return pd.read_csv(path)

def split_data(df, seed_random):
    X = df.drop(["id","date","zipcode", "price"], axis=1).values
    y = df["price"].values
    print("Datos totales:", X.shape, y.shape)
    return train_test_split(X, y, test_size=0.2, random_state=seed_random)

def transform_scaler(X, scaler):
    return scaler.transform(X)

def preprocess(X, path=SCALER_PATH):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, path)
    return X, scaler

def train(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    print("Alpha:", alpha)
    return model

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)

def eval_model(model, X, y):
    score = model.score(X,y)
    print("El R2 es: ", score)
    return score

def save_report(alpha, random_seed, R2 , path):
    metrics = {"alpha":alpha,
               "random_seed":random_seed,
               "R2": R2 }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def main():

    df = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_data(df, RANDOM_SEED)
    X_train, scaler = preprocess(X_train)
    print("Datos de entranamiento: ", X_train.shape, y_train.shape)
    print("Datos de prueba: ", X_test.shape, y_test.shape)
    model = train(X_train, y_train, alpha=ALPHA)
    save_model(model, MODEL_PATH)
    X_test = transform_scaler(X_test, scaler)
    metric = eval_model(model, X_test, y_test)
    save_report(ALPHA, RANDOM_SEED, metric, METRICS_PATH)

if __name__ == "__main__":
    main()

