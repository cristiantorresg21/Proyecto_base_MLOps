
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(data, seed_random,test_split):
    X = data.drop(["id","date","zipcode", "price"], axis=1).values
    y = data["price"].values
    print("Datos totales:", X.shape, y.shape)
    return train_test_split(X, y, test_size=test_split, random_state=seed_random)