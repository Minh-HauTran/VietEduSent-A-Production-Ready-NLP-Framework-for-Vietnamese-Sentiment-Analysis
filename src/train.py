from src.config import load_config
from src.preprocessing import clean_text
from src.vectorization import build_bow
from src.ml_models import get_model
from src.experiment import run_kfold
from src.tracking import start_experiment, log_metrics, log_params
from src.utils import set_seed

from datasets import load_dataset

def main():
    config = load_config()

    set_seed(config["project"]["seed"])

    run = start_experiment()

    dataset = load_dataset("uit-nlp/vietnamese_students_feedback")
    df = dataset["train"].to_pandas()

    df["text"] = df["sentence"].apply(clean_text)

    X_text = df["text"].values
    y = df["sentiment"].values

    vectorizer = build_bow(X_text)
    X = vectorizer.transform(X_text)

    model = get_model(config["model"]["type"])

    results = run_kfold(
        model,
        X,
        y,
        n_splits=config["experiment"]["kfold"]
    )

    log_metrics(results)
    log_params(config["model"])

    print("\n🔥 FINAL RESULTS:", results)

if __name__ == "__main__":
    main()
