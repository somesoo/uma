from typing import List, Tuple

Example = Tuple[int, str, str, str]

def load_dna_with_window(filepath: str, dataset_type: str, window_size: int = 5) -> List[Example]:
    examples = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        boundary_pos = int(lines[0].strip())
        lines = lines[1:]

        for i in range(0, len(lines), 2):
            label = int(lines[i].strip())
            full_seq = lines[i + 1].strip().upper()
            window_seq = full_seq[boundary_pos : boundary_pos + window_size]
            examples.append((label, window_seq, dataset_type, full_seq))
    return examples


p= load_dna_with_window("input_data/spliceATrainKIS.dat", "acceptor", 5)
print(p)