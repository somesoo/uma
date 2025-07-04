### Autor: Radosław Kasprzak

import numpy as np
from collections import Counter
from src.decision_tree.metrics_func import best_split
from src.decision_tree.tree_utils import split_dataset


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

class DecisionTree:
    def __init__(self, max_depth=None, min_samples=2, n_feats=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else self.n_feats
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_feats = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth) or (num_labels == 1) or (num_samples < self.min_samples):
            counts = Counter(y)
            total = sum(counts.values())
            p0 = counts.get(0, 0) / total
            p1 = counts.get(1, 0) / total
            return Node(value=Counter(y).most_common(1)[0][0], proba=[p0, p1])

        feat_idxs = range(num_feats)        
        best_feat, best_thr = best_split(X, y, feat_idxs, self.min_samples)
        if best_feat is None:
            counts = Counter(y); total = sum(counts.values())
            p0 = counts.get(0, 0) / total; p1 = counts.get(1, 0) / total
            return Node(value=Counter(y).most_common(1)[0][0], proba=[p0, p1])

        Xl, yl, Xr, yr = split_dataset(X, y, best_feat, best_thr)
        left = self._grow_tree(Xl, yl, depth + 1)
        right = self._grow_tree(Xr, yr, depth + 1)
        return Node(best_feat, best_thr, left, right)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict_proba(self, X):
        probs = []
        for x in X:
            node = self.root
            while node.value is None:
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            probs.append(node.proba)
        return np.array(probs)
    

    def export_text(self, feature_names=None):
        def recurse(node, prefix="", is_left=True):
            lines = []
            connector = "├── " if is_left else "└── "

            if node.value is not None:
                p0 = round(node.proba[0], 3)
                p1 = round(node.proba[1], 3)
                lines.append(f"{prefix}{connector}Predict: {node.value} (proba: [{p0}, {p1}])")
                return lines

            name = feature_names[node.feature_idx] if feature_names else f"x[{node.feature_idx}]"
            if node.threshold == 0.0:
                condition = f"{name} == 0"
                else_condition = f"{name} == 1"
            else:
                condition = f"{name} <= {node.threshold:.2f}"
                else_condition = f"{name} > {node.threshold:.2f}"

            lines.append(f"{prefix}{connector}If {condition}:")
            lines += recurse(node.left, prefix + ("│   " if is_left else "    "), True)
            lines.append(f"{prefix}{'│   ' if is_left else '    '}Else (if {else_condition}):")
            lines += recurse(node.right, prefix + ("│   " if is_left else "    "), False)
            return lines

        return "\n".join(recurse(self.root, "", True))


    @staticmethod
    def _most_common_label(y):
        return Counter(y).most_common(1)[0][0]
    