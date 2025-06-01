import argparse
from random import Random
from pathlib import Path

DNA_ALPHABET = "ACGT"

def generate_random_regex(k: int, max_wildcards: int, rng: Random) -> str:
    """Losowo tworzy regex z <= max_wildcards wildcardami, z preferencją środkowej liczby"""
    wildcard_count = rng.choices(
        population=list(range(max_wildcards + 1)),
        weights=[i + 1 for i in range(max_wildcards + 1)],  # bardziej trójkątny rozkład
        k=1
    )[0]

    base = [rng.choice(DNA_ALPHABET) for _ in range(k)]
    if wildcard_count > 0:
        wildcard_pos = rng.sample(range(k), wildcard_count)
        for i in wildcard_pos:
            base[i] = '.'
    return ''.join(base)

def generate_unique_random_patterns(k, max_wildcards, limit, existing: set, rng, try_limit=1_000_000):
    """Zwraca unikalne regexy z większą różnorodnością"""
    attempts = 0
    while len(existing) < limit and attempts < try_limit:
        regex = generate_random_regex(k, max_wildcards, rng)
        attempts += 1
        if regex not in existing:
            existing.add(regex)
            yield regex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True, help="długość wzorca")
    parser.add_argument("-w", "--wildcards", type=int, required=True, help="maks. liczba wildcardów '.'")
    parser.add_argument("-n", "--limit", type=int, required=True, help="docelowa liczba regexów")
    parser.add_argument("-o", "--output", default="input_data/regex_donor.txt", help="plik wynikowy")
    parser.add_argument("--seed", type=int, default=None, help="opcjonalne ziarno RNG")

    args = parser.parse_args()
    output_path = Path(args.output)

    rng = Random(args.seed) if args.seed is not None else Random()

    existing = set(output_path.read_text().splitlines()) if output_path.exists() else set()

    with output_path.open("a") as f:
        for regex in generate_unique_random_patterns(args.k, args.wildcards, args.limit, existing, rng):
            f.write(regex + "\n")

    print(f"Wygenerowano {len(existing):,} regexów z maks. {args.wildcards} wildcardami → {args.output}")

if __name__ == "__main__":
    main()