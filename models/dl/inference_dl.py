def predict_dl(model, vectorizer, texts):
    X = vectorizer(texts)
    preds = model.predict(X)
    return preds.argmax(axis=1)
