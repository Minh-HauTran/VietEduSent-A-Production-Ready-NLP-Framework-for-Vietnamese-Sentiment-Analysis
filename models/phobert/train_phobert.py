import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.phobert_model import build_phobert_model

MAX_LEN = 128

def encode_texts(tokenizer, texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

def train_phobert():
    df = pd.read_csv("data/processed/processed.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    tokenizer, model = build_phobert_model(MAX_LEN)

    train_enc = encode_texts(tokenizer, X_train)
    test_enc = encode_texts(tokenizer, X_test)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        [train_enc["input_ids"], train_enc["attention_mask"]],
        y_train,
        validation_data=(
            [test_enc["input_ids"], test_enc["attention_mask"]],
            y_test
        ),
        epochs=3,
        batch_size=16
    )

    return model, tokenizer
