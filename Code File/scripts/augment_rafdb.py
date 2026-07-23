from __future__ import annotations
import csv
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

ANGLES = [-15, -10, -5, 0, 5, 10, 15]

def augment_image(img: Image.Image) -> list[tuple[str, Image.Image]]:
    variants = []
    for angle in ANGLES:
        rotated = img.rotate(angle, fillcolor=0)
        variants.append((f"r{angle:+d}", rotated))
        flipped = rotated.transpose(Image.FLIP_LEFT_RIGHT)
        variants.append((f"r{angle:+d}_flip", flipped))
    return variants  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv',  required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--aug-subdir', default='aligned_aug')
    args = parser.parse_args()

    image_root = Path(args.image_root)
    output_csv = Path(args.output_csv)

    with open(args.train_csv, newline='') as f:
        source_rows = list(csv.DictReader(f))

    aug_rows = []
    failed = 0

    for row in tqdm(source_rows, desc="Augmenting"):
        src_path = image_root / row['image_path']
        if not src_path.exists():
            failed += 1
            continue

        img = Image.open(src_path)  

        for suffix, aug_img in augment_image(img):
            stem     = src_path.stem
            out_path = image_root / args.aug_subdir / row['expression'] / f"{stem}_{suffix}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            aug_img.save(out_path)

            aug_rows.append({
                'image_path': str(out_path.relative_to(image_root)),
                'expression': row['expression'],
                "identity": row["identity"], 
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'expression', 'identity'])
        writer.writeheader()
        writer.writerows(aug_rows)

    print(f"\nAugmentation complete:")
    print(f"  Source images: {len(source_rows)}")
    print(f"  Augmented rows: {len(aug_rows)}  (should be {len(source_rows)*14})")
    print(f"  Failed: {failed}")
    print(f"  Output CSV: {output_csv}")

if __name__ == '__main__':
    main()
    