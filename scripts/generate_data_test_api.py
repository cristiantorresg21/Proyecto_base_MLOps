import json
import pandas as pd
import numpy as np
from configs.paths import DATASET_PATH, TEST_API_REQUEST_PATH
from configs.configs import FEATURES
from src.data.load_data import load_data

# generar datos ejemplo (2 muestras, 3 features)
data = pd.read_csv(DATASET_PATH)[FEATURES]
data_test = data.sample(2, random_state=50).values.tolist()
#data_test = {"data": data_test}
print(data_test)
with open(TEST_API_REQUEST_PATH, "w") as f:
    json.dump(data_test, f, indent=4)

