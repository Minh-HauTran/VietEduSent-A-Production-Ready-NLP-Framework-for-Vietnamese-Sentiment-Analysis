from sklearn.model_selection import StratifiedKFold
from src.evaluate import compute_metrics

def run_kfold(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        metrics = compute_metrics(y_val, preds)
        results.append(metrics)

        print(metrics)

    avg = {
        key: sum([r[key] for r in results]) / len(results)
        for key in results[0]
    }

    print("\nAverage:", avg)
    return avg
