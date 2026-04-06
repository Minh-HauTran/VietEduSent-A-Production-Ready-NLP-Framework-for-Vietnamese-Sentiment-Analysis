def predict_phobert(model, tokenizer, texts, max_len=128):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )

    preds = model.predict([enc["input_ids"], enc["attention_mask"]])
    return preds.argmax(axis=1)
