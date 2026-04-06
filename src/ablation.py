from src.preprocessing import clean_text

def ablation_pipeline(texts):
    return {
        "raw": texts,
        "cleaned": [clean_text(t) for t in texts],
        "lower_only": [t.lower() for t in texts],
        "no_tokenize": [t.replace("_", " ") for t in texts],
    }
