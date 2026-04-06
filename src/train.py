from datasets import load_dataset
from src.preprocessing import clean_text
from src.vectorization import build_bow
from src.experiment import run_experiment
from src.config import load_config

def main():
    config = load_config()

    dataset = load_dataset("uit-nlp/vietnamese_students_feedback")
    df = dataset["train"].to_pandas()

    texts = df["sentence"].apply(clean_text).values
    labels = df["sentiment"].values

    vectorizer = build_bow(texts)
    X = vectorizer.transform(texts)

    run_experiment(X, labels, model_name=config["model"])

if __name__ == "__main__":
    main()
