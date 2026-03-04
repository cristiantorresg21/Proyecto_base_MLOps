import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib
import os
seed_random= 45
offset= "./Proyecto_base_MLOps"

def load_data(path: str):
    return pd.read_csv(path)

def split_data(df, seed_random):
    X = df.drop(["id","date","zipcode", "price"], axis=1).values
    y = df["price"].values.reshape(-1,1)
    print("Datos totales:", X.shape, y.shape)
    
    return train_test_split(X, y, random_state=seed_random)

def transform_scaler(X, scaler):
    return scaler.transform(X)

def preprocess(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, offset + "/artifacts/scaler.pkl")
    return X, scaler

def train(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def save_model(model, path= offset +"/artifacts/model.pkl"):
    joblib.dump(model, path)

def eval(model, X, y):
    print("El R2 es: ", model.score(X,y))
    return model.score(X,y)


def main():
    path= offset + "/data/house_data_washington.csv"

    df = load_data(path)

    X_train, X_test, y_train, y_test = split_data(df, seed_random)

    X_train, scaler = preprocess(X_train)
    
    print("Datos de entranamiento: ", X_train.shape, y_train.shape)
    print("Datos de prueba: ", X_test.shape, y_test.shape)

    model = train(X_train, y_train, alpha=0.01)

    save_model(model)

    X_test = transform_scaler(X_test, scaler)
    eval(model, X_test, y_test)

if __name__ == "__main__":
    main()

