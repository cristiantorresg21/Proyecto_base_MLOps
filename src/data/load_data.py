import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, features):
    data = pd.read_csv(path)
    X = data[features].values
    y = data["price"].values
    return X, y

def split_data(X, y, seed_random, test_split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed_random)
    return X_train, X_test, y_train, y_test