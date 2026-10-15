
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOTS = [
    ROOT / "datasets/india_traffic/train/labels",
    ROOT / "datasets/india_traffic/valid/labels",
]

for folder in LABEL_ROOTS:
    print(f"\nChecking: {folder}")
    counts = Counter()
    examples = {}

    for file in folder.glob("*.txt"):
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            if not parts:
                continue

            # Detection row: class + 4 box coordinates
            # Segmentation row: class + 6 or more coordinates
            row_type = (
                "detection" if len(parts) == 5
                else "segmentation" if len(parts) >= 7
                else "invalid"
            )
            counts[row_type] += 1
            examples.setdefault(row_type, (file.name, line_no, line))

    print("Row counts:", dict(counts))
    for kind, example in examples.items():
        print(f"{kind} example: {example}")
