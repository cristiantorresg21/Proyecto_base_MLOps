from sklearn.preprocessing import StandardScaler
import joblib

def transform_scaler(X, scaler):
    return scaler.transform(X)

def preprocess(X, path):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, path)
    return X, scaler