import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from src.ml_models import get_model
from src.evaluate import compute_metrics
from sklearn.model_selection import train_test_split

def train_ml_model(model_name="svm"):
    df = pd.read_csv("data/processed/processed.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = get_model(model_name)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    metrics = compute_metrics(y_test, preds)

    print(f"\n🔥 {model_name.upper()} Results:")
    print(metrics)

    return model, vectorizer
