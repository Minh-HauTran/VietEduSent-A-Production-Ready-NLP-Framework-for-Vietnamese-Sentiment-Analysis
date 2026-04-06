import json

def generate_report(metrics, path="reports/summary.txt"):
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
