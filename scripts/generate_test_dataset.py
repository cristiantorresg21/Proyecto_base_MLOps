import pathlib

import pandas as pd

from configs.paths import TEST_DATASET_DIR, DATASET_PATH
from src.data.load_data import load_data

data = pd.read_csv(DATASET_PATH)

### TESTS
TEST_DATASET_A_PATH = TEST_DATASET_DIR / "dataset_A.csv"
data_test = data.sample(10, random_state=50)
data_test.to_csv(TEST_DATASET_A_PATH, index=False)

TEST_DATASET_B_PATH = TEST_DATASET_DIR / "dataset_B.csv"
data_test = data.sample(20, random_state=100)
data_test.to_csv(TEST_DATASET_B_PATH, index=False)