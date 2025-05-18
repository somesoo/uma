from typing import List
from itertools import product

def save_kgram_regexes_to_file(k: int, output_file: str, alphabet: str = "ACGT") -> None:
    with open(output_file, "w") as f:
        for p in product(alphabet, repeat=k):
            f.write(''.join(p) + '\n')

def load_regex_patterns(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]
