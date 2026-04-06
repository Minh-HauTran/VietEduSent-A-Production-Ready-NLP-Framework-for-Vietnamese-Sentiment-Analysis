from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import TextVectorization
import numpy as np

def build_bow(texts):
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)
    return vectorizer

def build_dl_vectorizer(texts, max_vocab=20000):
    lengths = [len(t.split()) for t in texts]
    max_len = int(np.percentile(lengths, 95))

    vectorizer = TextVectorization(
        max_tokens=max_vocab,
        output_mode="int",
        output_sequence_length=max_len
    )

    vectorizer.adapt(texts)
    return vectorizer
