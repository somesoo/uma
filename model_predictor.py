import joblib
from typing import List
import re

MODEL_PATH = "decision_tree_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, sequences: List[str], regex_list: List[str], positions: List[int]) -> List[int]:
    X = []
    for seq in sequences:
        features = []
        for pos in positions:
            for regex in regex_list:
                if pos + len(regex) <= len(seq):
                    fragment = seq[pos:pos + len(regex)]
                    features.append(int(re.fullmatch(regex, fragment) is not None))
                else:
                    features.append(0)
        X.append(features)
    return model.predict(X).tolist()