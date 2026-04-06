import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.ml_models import get_model
from src.evaluate import evaluate

def run_experiment(X, y, model_name="svm", n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}")

        model = get_model(model_name)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = evaluate(y_val, preds)
        scores.append(acc)

    print(f"\nAverage Accuracy: {sum(scores)/len(scores):.4f}")
    return scores
