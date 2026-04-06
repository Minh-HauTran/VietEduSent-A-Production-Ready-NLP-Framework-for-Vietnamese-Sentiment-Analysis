import json
import os

def log_results(results, filename="results/logs.json"):
    os.makedirs("results", exist_ok=True)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(results)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
