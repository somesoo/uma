from typing import List, Tuple

Example = Tuple[int, str, str, str]

def load_dna_with_window(filepath: str, dataset_type: str, regex_len: str) -> List[Example]:
    examples = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        boundary_pos = int(lines[0].strip())
        lines = lines[1:]

        for i in range(0, len(lines), 2):
            label = int(lines[i].strip())
            full_seq = lines[i + 1].strip().upper()
            if dataset_type == "donor":
                full_seq = full_seq[:7] + full_seq[9:]
            elif dataset_type == "acceptor":
                full_seq = full_seq[:68] + full_seq[70:]
            start = boundary_pos
            end = boundary_pos + regex_len
            if end <= len(full_seq):  # zabezpieczenie
                #print(full_seq)
                window_seq = full_seq[start:end]
                #print(window_seq)
                examples.append((label, window_seq, dataset_type, full_seq))
            else:
                print(f"Ostrzeżenie: sekwencja z etykietą {label} za krótka (len={len(full_seq)}), pominięto.")

    return examples