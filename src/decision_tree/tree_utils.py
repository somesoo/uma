def split_dataset(X, y, feature_idx, threshold):
    mask = X[:, feature_idx] <= threshold
    return X[mask], y[mask], X[~mask], y[~mask]