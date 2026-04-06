import numpy as np
from sklearn.utils import resample

def oversample_data(X, y):
    neutral_idx = np.where(y == 1)[0]

    target_size = max(
        len(y[y == 0]),
        len(y[y == 2])
    )

    oversampled_idx = resample(
        neutral_idx,
        replace=True,
        n_samples=target_size
    )

    X_new = np.concatenate([X, X[oversampled_idx]])
    y_new = np.concatenate([y, y[oversampled_idx]])

    return X_new, y_new
