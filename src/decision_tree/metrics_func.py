import numpy as np
from collections import Counter


def gini_impurity(y):
    m = len(y)
    if m == 0:
        return 0
    counts = np.array(list(Counter(y).values()))
    probs = counts / m
    return 1 - np.sum(probs ** 2)


def best_split(X, y, feature_idxs, min_samples=2):
    from src.decision_tree.tree_utils import split_dataset
    best_gini = float('inf')
    best_feat, best_thr = None, None
    for idx in feature_idxs:
        thresholds = np.unique(X[:, idx])
        for thr in thresholds:
            Xl, yl, Xr, yr = split_dataset(X, y, idx, thr)
            if len(yl) < min_samples or len(yr) < min_samples:
                continue
            from src.decision_tree.metrics_func import gini_impurity
            g_left = gini_impurity(yl)
            g_right = gini_impurity(yr)
            w_left = len(yl) / len(y)
            w_right = len(yr) / len(y)
            g_total = w_left * g_left + w_right * g_right
            if g_total < best_gini:
                best_gini, best_feat, best_thr = g_total, idx, thr
    return best_feat, best_thr
