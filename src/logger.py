import json
import os

def log_results(result, path="results/logs.json"):
    os.makedirs("results", exist_ok=True)

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
