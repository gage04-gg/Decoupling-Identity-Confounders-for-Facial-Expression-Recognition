from __future__ import annotations
import csv
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

RAFDB_LABELS = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happy',
    '5': 'sadness',
    '6': 'anger',
    '7': 'neutral',
}


def whiten(img_array: np.ndarray) -> np.ndarray:
    img = img_array.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std < 1e-6:
        std = 1.0
    whitened = (img - mean) / std
    whitened = whitened - whitened.min()
    max_val = whitened.max()
    if max_val > 0:
        whitened = whitened / max_val * 255.0
    return whitened.astype(np.uint8)


def process_split(
    image_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    split: str,
    apply_whitening: bool = True,
) -> list[dict]:
    rows = []
    failed = 0

    with open(labels_csv, newline='') as f:
        label_rows = list(csv.DictReader(f))

    for i, row in enumerate(tqdm(label_rows, desc=f"Processing {split}")):
        filename = row['image'].strip()
        label_code = row['label'].strip()

        if label_code not in RAFDB_LABELS:
            print(f"  Unknown label {label_code} for {filename}, skipping")
            continue

        expression = RAFDB_LABELS[label_code]
        src_path = image_dir / filename

        if not src_path.exists():
            print(f"  Missing: {src_path}")
            failed += 1
            continue

        try:
            img = Image.open(src_path).convert('RGB')
            img = img.resize((112, 112), Image.BICUBIC)
            img_gray = np.array(img.convert('L'))
            if apply_whitening:
                img_gray = whiten(img_gray)
            stem = Path(filename).stem  
            out_path = output_dir / 'aligned' / split / expression / f"{stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_gray, mode='L').save(out_path)
            identity = stem  
            rows.append({
                'image_path': str(out_path.relative_to(output_dir)),
                'expression': expression,
                'identity': identity,
                'split': split,
            })

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            failed += 1

    print(f"  Done: {len(rows)} processed, {failed} failed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root',    required=True,
                        help='data_raw/RAF-DB — contains train/, test/, train_labels.csv, test_labels.csv')
    parser.add_argument('--output-root', required=True,
                        help='data_processed/RAF-DB')
    parser.add_argument('--no-whitening', action='store_true')
    args = parser.parse_args()

    raw_root   = Path(args.raw_root)
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_whitening = not args.no_whitening

    # Process train split
    train_rows = process_split(
        image_dir    = raw_root / 'train',
        labels_csv   = raw_root / 'train_labels.csv',
        output_dir   = output_dir,
        split        = 'train',
        apply_whitening = apply_whitening,
    )

    # Process test split
    test_rows = process_split(
        image_dir    = raw_root / 'test',
        labels_csv   = raw_root / 'test_labels.csv',
        output_dir   = output_dir,
        split        = 'test',
        apply_whitening = apply_whitening,
    )

    meta_dir = output_dir / 'metadata'
    meta_dir.mkdir(exist_ok=True)

    fieldnames = ['image_path', 'expression', 'identity']

    train_csv = meta_dir / 'train.csv'
    with open(train_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in train_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nTrain CSV: {train_csv} ({len(train_rows)} rows)")

    test_csv = meta_dir / 'test.csv'
    with open(test_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in test_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Test CSV:  {test_csv} ({len(test_rows)} rows)")

    # Write summary
    from collections import Counter
    train_dist = Counter(r['expression'] for r in train_rows)
    test_dist  = Counter(r['expression'] for r in test_rows)

    summary = {
        'train_total': len(train_rows),
        'test_total':  len(test_rows),
        'train_distribution': dict(train_dist),
        'test_distribution':  dict(test_dist),
        'num_classes': len(train_dist),
        'image_size': '112x112',
        'channels': 1,
        'whitening': apply_whitening,
    }
    (meta_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nSummary:")
    print(f"  Train: {len(train_rows)} images, {len(train_dist)} classes")
    print(f"  Test:  {len(test_rows)} images")
    print(f"  Train distribution: {dict(sorted(train_dist.items()))}")


if __name__ == '__main__':
    main()
    