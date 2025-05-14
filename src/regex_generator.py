from itertools import product
from typing import List

def generate_kgram_regexes(k: int, alphabet: str = "ACGT") -> List[str]:
    return [''.join(p) for p in product(alphabet, repeat=k)]
