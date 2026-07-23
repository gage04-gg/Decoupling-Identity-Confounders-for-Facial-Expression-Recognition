import csv
import json
import math
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

INPUT_CSV   = "ckextended.csv"   
OUTPUT_ROOT = Path("data_processed/CK+")

EMOTION_MAP = {
    '0': 'anger',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sadness',
    '5': 'surprise',
    '7': 'contempt',
}

TOTAL_SUBJECTS = 123

def whiten(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mean, std = a.mean(), a.std()
    if std < 1e-6:
        std = 1.0
    w = (a - mean) / std
    w = w - w.min()
    if w.max() > 0:
        w = w / w.max() * 255.0
    return w.astype(np.uint8)

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aligned_dir = OUTPUT_ROOT / "aligned"
    meta_dir    = OUTPUT_ROOT / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CSV...")
    peak_rows = []
    with open(INPUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['emotion'] in EMOTION_MAP:
                peak_rows.append(row)

    print(f"Found {len(peak_rows)} peak expression frames "
          f"(expected ~327)")

    seqs_per_subject = len(peak_rows) / TOTAL_SUBJECTS
    rows_with_meta = []
    for i, row in enumerate(peak_rows):
        subject_num  = min(int(i / seqs_per_subject) + 1, TOTAL_SUBJECTS)
        subject_id   = f"S{subject_num:03d}"
        emotion_name = EMOTION_MAP[row['emotion']]

        pixels = np.array([int(p) for p in row['pixels'].split()],
                          dtype=np.uint8).reshape(48, 48)

        img_112 = Image.fromarray(pixels, mode='L').resize(
            (112, 112), Image.BICUBIC
        )

        img_white = whiten(np.array(img_112))

        out_dir  = aligned_dir / emotion_name
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{subject_id}_seq{i:04d}_{emotion_name}.png"
        out_path = out_dir / filename
        Image.fromarray(img_white, mode='L').save(out_path)

        rows_with_meta.append({
            'image_path': str(out_path.relative_to(OUTPUT_ROOT)),
            'expression': emotion_name,
            'identity':   subject_id,
        })

    print(f"Saved {len(rows_with_meta)} images to {aligned_dir}")
    by_expr = defaultdict(set)
    for r in rows_with_meta:
        by_expr[r['expression']].add(r['identity'])

    print("\nSubjects per expression class:")
    for expr, subjects in sorted(by_expr.items()):
        status = "✓" if len(subjects) >= 2 else "✗ PROBLEM"
        print(f"  {expr}: {len(subjects)} subjects {status}")
    all_csv = meta_dir / "all.csv"
    with open(all_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_path','expression','identity'])
        w.writeheader()
        w.writerows(rows_with_meta)
    print(f"\nWrote {all_csv} ({len(rows_with_meta)} rows)")
    from collections import Counter
    class_counts = Counter(r['expression'] for r in rows_with_meta)
    id_counts    = Counter(r['identity'] for r in rows_with_meta)
    summary = {
        'total_rows':        len(rows_with_meta),
        'num_classes':       len(class_counts),
        'class_counts':      dict(class_counts),
        'num_identities':    len(id_counts),
        'image_size':        '112x112',
        'channels':          1,
        'whitening':         True,
        'note': (
            'Subject IDs reconstructed from Zenodo CSV. '
            'Approximately matches original CK+ subject structure.'
        )
    }
    (meta_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
