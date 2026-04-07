from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    # 🔬 Robustness Evaluation: Noisy Vietnamese Text
def evaluate_noisy_samples(model):
    noisy_samples = [
        "thầy dạy hay quáaaaa",
        "ko hiểu bài luôn :((((",
        "giảng viên rất tốt 👍👍",
        "bài tập nhìu quáaaa",
        "hok hiểu j hết trơn"
    ]

    print("\n🧪 Robustness Test (Noisy Inputs):")
    for text in noisy_samples:
        pred = model.predict([text])[0]
        print(f"Input: {text} → Prediction: {pred}")

    # Insight:
    # Vietnamese user-generated text contains slang, emoji, and noise
    # Robust preprocessing is critical for stable real-world performance
