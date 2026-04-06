import pickle

def predict(texts, model_path, vectorizer_path):
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    X = vectorizer.transform(texts)
    return model.predict(X)
