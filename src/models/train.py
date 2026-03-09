from sklearn.linear_model import Ridge
import joblib

def train_model(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)