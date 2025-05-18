import argparse
from itertools import product, combinations
from random import sample, Random
from pathlib import Path
DNA_ALPHABET = "ACGT"

def generate_regexes(k: int, max_wildcards: int):
    """Generuje wszystkie regexy długości k z <= max_wildcards kropkami '.'"""
    idx_range = range(k)
    for base in product(DNA_ALPHABET, repeat=k):
        base = list(base)
        yield ''.join(base)  # bez wildcardów

        for w in range(1, max_wildcards + 1):
            for pos in combinations(idx_range, w):
                temp = base.copy()
                for i in pos:
                    temp[i] = '.'
                yield ''.join(temp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True, help="długość wzorca")
    parser.add_argument("-w", "--wildcards", type=int, default=0,
                        help="maks. liczba wildcardów '.' w pojedynczym wzorcu")
    parser.add_argument("-n", "--limit", type=int,
                        help="opcjonalnie: ogranicz liczbę wyników (losowo)")
    parser.add_argument("-o", "--output", required=False, default="input_data/regex_patterns.txt",
                        help="plik wynikowy (np. regex.txt)")

    args = parser.parse_args()

    # Generowanie pełnej puli regexów
    all_patterns = list(generate_regexes(args.k, args.wildcards))

    # Jeśli podano limit - losowy podzbiór
    if args.limit:
        rnd = Random(2025)
        patterns = sample(all_patterns, min(args.limit, len(all_patterns)))
    else:
        patterns = all_patterns

    # Zapis do pliku
    Path(args.output).write_text('\n'.join(patterns))
    print(f"Wygenerowano {len(patterns):,} wzorców do pliku: {args.output}")

if __name__ == "__main__":
    main()