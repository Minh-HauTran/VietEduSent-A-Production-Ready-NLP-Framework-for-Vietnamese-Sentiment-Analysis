from src.preprocessing import clean_text

def ablation_preprocessing(texts):
    versions = {
        "raw": texts,
        "cleaned": [clean_text(t) for t in texts],
        "no_tokenize": [t.lower() for t in texts],
    }
    return versions
