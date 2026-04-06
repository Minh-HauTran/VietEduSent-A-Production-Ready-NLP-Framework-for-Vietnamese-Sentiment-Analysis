import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.vectorization import build_dl_vectorizer
from src.dl_models import build_gru

def train_dl():
    df = pd.read_csv("data/processed/processed.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    vectorizer = build_dl_vectorizer(X_train)

    X_train_vec = vectorizer(X_train)
    X_test_vec = vectorizer(X_test)

    model = build_gru(vocab_size=len(vectorizer.get_vocabulary()))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_vec,
        y_train,
        validation_data=(X_test_vec, y_test),
        epochs=5,
        batch_size=64
    )

    return model, vectorizer
