import numpy as np
from sklearn.utils import resample
from collections import Counter

def oversampling(X, y, rd):
    X = np.array(X)
    y = np.array(y)

    mask_pos = (y == 1)
    mask_neg = (y == 0)

    X_pos = X[mask_pos]
    y_pos = y[mask_pos]

    X_neg = X[mask_neg]
    y_neg = y[mask_neg]

    # Jeśli klasa 1 jest mniejszością -> oversampling do rozmiaru klasy 0
    if len(y_pos) < len(y_neg) and len(y_pos) > 0:
        X_pos_res, y_pos_res = resample(
            X_pos, y_pos,
            replace=True,
            n_samples=len(y_neg),
            random_state=rd
        )
        X_balanced = np.vstack([X_neg, X_pos_res])
        y_balanced = np.concatenate([y_neg, y_pos_res])
    elif len(y_neg) < len(y_pos) and len(y_neg) > 0:
        # Jeśli klasa 0 jest mniejszością -> oversampling do rozmiaru klasy 1
        X_neg_res, y_neg_res = resample(
            X_neg, y_neg,
            replace=True,
            n_samples=len(y_pos),
            random_state=rd
        )
        X_balanced = np.vstack([X_pos, X_neg_res])
        y_balanced = np.concatenate([y_pos, y_neg_res])
    else:
        # Już zbalansowane albo brak próbek do oversamplingu
        X_balanced = X
        y_balanced = y

    # Mieszamy kolejność po oversamplingu
    perm = np.random.RandomState(rd).permutation(len(y_balanced))
    X_balanced = X_balanced[perm]
    y_balanced = y_balanced[perm]

    X, y = X_balanced, y_balanced
    print("Po balansowaniu:", Counter(y))

