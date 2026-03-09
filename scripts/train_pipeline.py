import argparse

from src.data.load_data import load_data, split_data
from src.preprocess.preprocess import preprocess, transform_scaler
from src.models.train import train_model, save_model
from src.evaluate.evaluate import eval_model, save_report
from configs.paths import DATASET_PATH, MODEL_PATH, SCALER_PATH, METRICS_PATH
from configs.configs import FEATURES

parser = argparse.ArgumentParser(description='Modelo de regresión lineal con regularización Ridge')
parser.add_argument('--alpha', '-a', default=0.0000001, type=float, help='El parametro alpha del modelo Ridge')
parser.add_argument('--seed', '-s' , default=45, type=int, help='Semilla de los numeros aleatorios')
parser.add_argument('--test_split', '-t' , default=0.2, type=float, help='Tamaño en porcentaje del dataset test')
args = parser.parse_args()
ALPHA = args.alpha
RANDOM_SEED = args.seed
TEST_SPLIT = args.test_split

def main():
    X, y = load_data(path=DATASET_PATH, features=FEATURES)
    X_train, X_test, y_train, y_test = split_data(X, y, seed_random=RANDOM_SEED, test_split=TEST_SPLIT)
    X_train, scaler = preprocess(X_train, path=SCALER_PATH)
    print("Datos de entranamiento: ", X_train.shape, y_train.shape)
    print("Datos de prueba: ", X_test.shape, y_test.shape)
    model = train_model(X_train, y_train, alpha=ALPHA)
    save_model(model, path=MODEL_PATH)
    X_test = transform_scaler(X_test, scaler)
    metric = eval_model(model, X_test, y_test)
    save_report(alpha=ALPHA, random_seed=RANDOM_SEED, R2=metric, split_test=TEST_SPLIT, path=METRICS_PATH)

if __name__ == "__main__":
    main()
