from src.preprocessing import clean_text

def run_ablation(texts):
    return {
        "raw": texts,
        "cleaned": [clean_text(t) for t in texts],
        "lower_only": [t.lower() for t in texts]
    }
