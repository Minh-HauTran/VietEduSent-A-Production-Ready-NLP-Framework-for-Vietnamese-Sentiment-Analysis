import pandas as pd
import os
from src.preprocessing import clean_text

def build_processed_data(
    input_path="data/raw/raw_data.csv",
    output_path="data/processed/processed.csv"
):
    df = pd.read_csv(input_path)

    df = df.dropna()
    df = df.drop_duplicates("sentence")

    df["text"] = df["sentence"].apply(clean_text)

    df.rename(columns={"sentiment": "label"}, inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ Processed dataset saved")

if __name__ == "__main__":
    build_processed_data()
