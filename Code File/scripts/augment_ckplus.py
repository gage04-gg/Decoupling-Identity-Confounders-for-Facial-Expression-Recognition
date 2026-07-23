import csv
import random
import argparse
from pathlib import Path
from PIL import Image

ANGLES = [-15, -10, -5, 0, 5, 10, 15]

def augment(img):
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
    args = parser.parse_args()

    image_root = Path(args.image_root)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_csv, newline='') as f:
        source_rows = list(csv.DictReader(f))

    aug_rows = []
    failed  = 0

    for row in source_rows:
        src = image_root / row['image_path']
        if not src.exists():
            print(f"Missing: {src}")
            failed += 1
            continue

        img = Image.open(src)
        for suffix, aug_img in augment(img):
            out = image_root / 'aligned_aug_ckplus' / row['expression'] / f"{src.stem}_{suffix}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            aug_img.save(out)
            aug_rows.append({
                'image_path': str(out.relative_to(image_root)),
                'expression': row['expression'],
                'identity':   row['identity'] + '_' + suffix,
            })

    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_path','expression','identity'])
        w.writeheader()
        w.writerows(aug_rows)

    print(f"Done: {len(aug_rows)} augmented rows "
          f"(expected {len(source_rows)*14}), {failed} failed")
    print(f"Output: {args.output_csv}")

if __name__ == '__main__':
    main()

