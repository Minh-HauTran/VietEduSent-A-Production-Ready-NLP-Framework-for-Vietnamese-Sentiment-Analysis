from datasets import load_dataset
from src.preprocessing import clean_text
from src.vectorization import build_bow
from src.ml_models import get_model
from src.experiment import run_kfold
from src.utils import set_seed

def load_data():
    dataset = load_dataset("uit-nlp/vietnamese_students_feedback")

    df = dataset["train"].to_pandas()
    df = df.dropna()
    df = df.drop_duplicates("sentence")

    df["text"] = df["sentence"].apply(clean_text)

    return df

def main():
    set_seed(42)

    df = load_data()

    texts = df["text"].values
    labels = df["sentiment"].values

    vectorizer = build_bow(texts)
    X = vectorizer.transform(texts)

    model = get_model("svm")

    results = run_kfold(model, X, labels)

    print("\nFinal Results:", results)

if __name__ == "__main__":
    main()
