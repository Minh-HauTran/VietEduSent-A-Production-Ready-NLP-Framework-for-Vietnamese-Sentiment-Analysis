from datasets import load_dataset
import pandas as pd
import os

def download_dataset(save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset("uit-nlp/vietnamese_students_feedback")

    df = pd.concat([
        dataset["train"].to_pandas(),
        dataset["validation"].to_pandas(),
        dataset["test"].to_pandas()
    ])

    df.to_csv(f"{save_path}/raw_data.csv", index=False)
    print("✅ Raw data saved")

if __name__ == "__main__":
    download_dataset()
