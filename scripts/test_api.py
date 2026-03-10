
import json

import requests

from configs.paths import TEST_API_REQUEST_PATH

with open(TEST_API_REQUEST_PATH, "r") as f:
    data = json.load(f)

url = "http://localhost:8000"
endpoint = "/predict"
response = requests.post(url+endpoint, json=data)
print(response)
print(response.json(), type(response.json()))

