import csv
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--output-dir',   required=True)
    parser.add_argument('--num-folds',    type=int, default=10)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    print(f"Total rows: {len(rows)}")

    subject_to_rows = defaultdict(list)
    for row in rows:
        subject_to_rows[row['identity']].append(row)

    subjects = sorted(subject_to_rows.keys())
    random.shuffle(subjects)
    print(f"Total subjects: {len(subjects)}")

    n = args.num_folds
    fold_subjects = [[] for _ in range(n)]
    for i, subj in enumerate(subjects):
        fold_subjects[i % n].append(subj)

    fieldnames = ['image_path', 'expression', 'identity']

    fold_stats = []
    for fold_idx in range(n):
        fold_num = f"{fold_idx + 1:02d}"

        val_subjects   = set(fold_subjects[fold_idx])
        train_subjects = set(subjects) - val_subjects

        val_rows   = [r for r in rows if r['identity'] in val_subjects]
        train_rows = [r for r in rows if r['identity'] in train_subjects]
        train_classes = set(r['expression'] for r in train_rows)
        val_classes   = set(r['expression'] for r in val_rows)

        print(f"Fold {fold_num}: train={len(train_rows)} "
              f"({len(train_subjects)} subjects), "
              f"val={len(val_rows)} ({len(val_subjects)} subjects)")
        print(f"  Train classes: {sorted(train_classes)}")
        print(f"  Val classes:   {sorted(val_classes)}")

        train_csv = output_dir / f"fold_{fold_num}_train.csv"
        with open(train_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in train_rows:
                w.writerow({k: r[k] for k in fieldnames})

        val_csv = output_dir / f"fold_{fold_num}_val.csv"
        with open(val_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in val_rows:
                w.writerow({k: r[k] for k in fieldnames})

        fold_stats.append({
            'fold': int(fold_num),
            'train_rows': len(train_rows),
            'val_rows': len(val_rows),
            'train_subjects': len(train_subjects),
            'val_subjects': len(val_subjects),
        })

    (output_dir / 'fold_summary.json').write_text(
        json.dumps(fold_stats, indent=2)
    )
    print(f"\nFolds written to {output_dir}")

if __name__ == '__main__':
    main()
