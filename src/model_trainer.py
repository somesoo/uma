import re
import joblib
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from typing import List, Tuple

Example = Tuple[int, str, str, str]

def extract_features(examples: List[Example], regex_list: List[str], positions: List[int]) -> Tuple[List[List[int]], List[int]]:
    X, y = [], []
    for label, full_seq, *_ in examples:
        features = []
        for pos in positions:
            for regex in regex_list:
                if pos + len(regex) <= len(full_seq):
                    fragment = full_seq[pos:pos + len(regex)]
                    features.append(int(re.fullmatch(regex, fragment) is not None))
                else:
                    features.append(0)
        X.append(features)
        y.append(label)
    return X, y

def train_and_save_model(X, y, model_path: str):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    print("Validation score:", clf.score(X_val, y_val))
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    