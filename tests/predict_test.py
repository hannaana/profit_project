import json
import os
import requests


current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)

# json_file_path = current_path + "/../NewLineProfit.json"
json_file_path = os.path.join(current_path, "..", "NewLineProfit.json")
print(json_file_path)

with open(json_file_path, 'r') as json_file:
    payload = json.load(json_file)

# print(payload)

data = json.dumps(payload)
print(data)
API_ENDPOINT = "http://127.0.0.1:5000/predict"

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)
# print(r)
# extracting response text
result = r.text
print(result)

