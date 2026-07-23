from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

CKPLUS_LABELS = {
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise",
}

RAFDB_LABELS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

AFFECTNET_LABELS = {
    0: "neutral",
    1: "happy",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
    7: "contempt",
}

OULU_EXPRESSIONS = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
}


@dataclass(frozen=True)
class RawSample:
    image_path: Path
    expression: str
    identity: str
    split: str


@dataclass(frozen=True)
class ProcessedSample:
    image_path: str
    expression: str
    identity: str
    split: str


class FaceAligner:
    def __init__(self, image_size: int = 112, device: str = "cpu") -> None:
        from facenet_pytorch import MTCNN

        self.image_size = image_size
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
        self.template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def align_to_grayscale(self, image_path: Path) -> Image.Image | None:
        image = Image.open(image_path).convert("RGB")
        boxes, _, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or landmarks is None or len(boxes) == 0:
            return None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        face_index = int(np.argmax(areas))
        src = landmarks[face_index].astype(np.float32)

        transform, _ = cv2.estimateAffinePartial2D(src, self.template, method=cv2.LMEDS)
        if transform is None:
            return None

        rgb = np.array(image)
        aligned = cv2.warpAffine(
            rgb,
            transform,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(gray)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FER datasets for DICE-FER.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ck = subparsers.add_parser("ckplus", help="Prepare CK+ from cohn-kanade-images and Emotion folders.")
    ck.add_argument("--images-root", required=True)
    ck.add_argument("--emotion-root", required=True)
    ck.add_argument("--output-root", required=True)
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--no-whitening", action="store_true", help="Disable paper-style per-image whitening.")

    oulu = subparsers.add_parser("oulu", help="Prepare Oulu-CASIA by taking last 3 frames per sequence.")
    oulu.add_argument("--raw-root", required=True)
    oulu.add_argument("--output-root", required=True)
    oulu.add_argument("--device", default="cpu")
    oulu.add_argument("--no-whitening", action="store_true", help="Disable paper-style per-image whitening.")

    raf = subparsers.add_parser("rafdb", help="Prepare RAF-DB basic-expression split.")
    raf.add_argument("--images-root", required=True, help="Folder containing aligned/original RAF images.")
    raf.add_argument("--labels-file", required=True, help="RAF list_patition_label.txt.")
    raf.add_argument("--output-root", required=True)
    raf.add_argument("--device", default="cpu")
    raf.add_argument("--no-whitening", action="store_true", help="Disable paper-style per-image whitening.")

    aff = subparsers.add_parser("affectnet", help="Prepare AffectNet seven-class train/validation metadata.")
    aff.add_argument("--images-root", required=True)
    aff.add_argument("--annotations", required=True, help="AffectNet CSV annotation file.")
    aff.add_argument("--output-root", required=True)
    aff.add_argument("--path-col", default="subDirectory_filePath")
    aff.add_argument("--expression-col", default="expression")
    aff.add_argument("--split", choices=["train", "val"], required=True)
    aff.add_argument("--device", default="cpu")
    aff.add_argument("--no-whitening", action="store_true", help="Disable paper-style per-image whitening.")

    generic = subparsers.add_parser("align-csv", help="Align images from a CSV with image_path,expression,identity,split.")
    generic.add_argument("--input-csv", required=True)
    generic.add_argument("--image-root", required=True)
    generic.add_argument("--output-root", required=True)
    generic.add_argument("--device", default="cpu")
    generic.add_argument("--no-whitening", action="store_true", help="Disable paper-style per-image whitening.")

    folds = subparsers.add_parser("make-folds", help="Create subject-wise 10-fold train/val CSVs.")
    folds.add_argument("--metadata-csv", required=True)
    folds.add_argument("--output-dir", required=True)
    folds.add_argument("--num-folds", type=int, default=10)
    folds.add_argument("--seed", type=int, default=42)

    augment = subparsers.add_parser("augment-csv", help="Create the paper's 14 deterministic rotation/flip views.")
    augment.add_argument("--input-csv", required=True)
    augment.add_argument("--image-root", required=True)
    augment.add_argument("--output-csv", required=True)
    augment.add_argument("--output-subdir", default="aligned_paper_aug")

    validate = subparsers.add_parser("validate-csv", help="Validate paths, classes, identities, and pairability.")
    validate.add_argument("--metadata-csv", required=True)
    validate.add_argument("--image-root", required=True)

    args = parser.parse_args()

    if args.command == "ckplus":
        samples = collect_ckplus(Path(args.images_root), Path(args.emotion_root))
        process_samples(samples, Path(args.output_root), args.device, whitening=not args.no_whitening)
    elif args.command == "oulu":
        samples = collect_oulu(Path(args.raw_root))
        process_samples(samples, Path(args.output_root), args.device, whitening=not args.no_whitening)
    elif args.command == "rafdb":
        samples = collect_rafdb(Path(args.images_root), Path(args.labels_file))
        process_samples(samples, Path(args.output_root), args.device, whitening=not args.no_whitening)
    elif args.command == "affectnet":
        samples = collect_affectnet(
            Path(args.images_root),
            Path(args.annotations),
            args.path_col,
            args.expression_col,
            args.split,
        )
        process_samples(samples, Path(args.output_root), args.device, whitening=not args.no_whitening)
    elif args.command == "align-csv":
        samples = collect_generic_csv(Path(args.input_csv), Path(args.image_root))
        process_samples(samples, Path(args.output_root), args.device, whitening=not args.no_whitening)
    elif args.command == "make-folds":
        make_subject_folds(Path(args.metadata_csv), Path(args.output_dir), args.num_folds, args.seed)
    elif args.command == "augment-csv":
        augment_csv(
            input_csv=Path(args.input_csv),
            image_root=Path(args.image_root),
            output_csv=Path(args.output_csv),
            output_subdir=args.output_subdir,
        )
    elif args.command == "validate-csv":
        validate_csv(Path(args.metadata_csv), Path(args.image_root))


def collect_ckplus(images_root: Path, emotion_root: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    for label_path in sorted(emotion_root.rglob("*.txt")):
        rel = label_path.relative_to(emotion_root)
        if len(rel.parts) < 3:
            continue

        identity = rel.parts[0]
        sequence = rel.parts[1]
        label_value = int(float(label_path.read_text().strip()))
        expression = CKPLUS_LABELS.get(label_value)
        if expression is None or expression == "contempt":
            continue

        frames_dir = images_root / identity / sequence
        frames = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(frames) < 2:
            continue

        samples.append(RawSample(frames[0], "neutral", identity, "all"))
        samples.append(RawSample(frames[-1], expression, identity, "all"))

    return samples


def collect_oulu(raw_root: Path) -> list[RawSample]:
    sequence_groups: dict[tuple[str, str, Path], list[Path]] = {}
    for image_path in sorted(raw_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        expression = infer_expression_from_path(image_path)
        identity = infer_identity_from_path(image_path)
        if expression is None or identity is None:
            continue
        sequence_dir = image_path.parent
        sequence_groups.setdefault((identity, expression, sequence_dir), []).append(image_path)

    samples: list[RawSample] = []
    for (identity, expression, _), frames in sequence_groups.items():
        for frame in sorted(frames)[-3:]:
            samples.append(RawSample(frame, expression, identity, "all"))
    return samples


def collect_rafdb(images_root: Path, labels_file: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    with labels_file.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, label_text = parts
            expression = RAFDB_LABELS[int(label_text)]
            split = "train" if filename.startswith("train") else "val"
            image_path = find_image(images_root, filename)
            if image_path is None:
                continue
            samples.append(RawSample(image_path, expression, Path(filename).stem, split))
    return samples


def collect_affectnet(
    images_root: Path,
    annotations: Path,
    path_col: str,
    expression_col: str,
    split: str,
) -> list[RawSample]:
    samples: list[RawSample] = []
    with annotations.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(float(row[expression_col]))
            expression = AFFECTNET_LABELS.get(label)
            if expression is None or expression == "contempt":
                continue
            rel_path = Path(row[path_col])
            image_path = images_root / rel_path
            if not image_path.exists():
                continue
            samples.append(RawSample(image_path, expression, rel_path.stem, split))
    return samples


def collect_generic_csv(input_csv: Path, image_root: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = Path(row["image_path"])
            if not path.is_absolute():
                path = image_root / path
            samples.append(
                RawSample(
                    image_path=path,
                    expression=row["expression"].strip(),
                    identity=row["identity"].strip(),
                    split=row.get("split", "all").strip() or "all",
                )
            )
    return samples


def process_samples(samples: Iterable[RawSample], output_root: Path, device: str, whitening: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    aligned_dir = output_root / "aligned"
    metadata_dir = output_root / "metadata"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    aligner = FaceAligner(device=device)
    processed: list[ProcessedSample] = []
    failed: list[str] = []

    input_samples = list(samples)
    for sample in tqdm(input_samples, desc="align faces"):
        output_rel = build_output_relative_path(sample)
        output_path = aligned_dir / output_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aligned = aligner.align_to_grayscale(sample.image_path)
        if aligned is None:
            failed.append(str(sample.image_path))
            continue
        if whitening:
            aligned = whiten_grayscale_image(aligned)
        aligned.save(output_path)

        processed.append(
            ProcessedSample(
                image_path=str(Path("aligned") / output_rel),
                expression=sample.expression,
                identity=sample.identity,
                split=sample.split,
            )
        )

    splits = sorted({sample.split for sample in processed})
    if splits == ["all"]:
        write_metadata_csv(metadata_dir / "all.csv", processed)
    else:
        for split in splits:
            write_metadata_csv(metadata_dir / f"{split}.csv", [sample for sample in processed if sample.split == split])
        merge_split_metadata(metadata_dir)

    if failed:
        (metadata_dir / "failed_alignment.txt").write_text("\n".join(failed) + "\n")
    write_summary_json(metadata_dir / "summary.json", processed, failed, attempted=len(input_samples))
    print(f"wrote {len(processed)} aligned samples to {aligned_dir}")
    print(f"failed to align {len(failed)} samples")
    print(f"wrote summary to {metadata_dir / 'summary.json'}")


def make_subject_folds(metadata_csv: Path, output_dir: Path, num_folds: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata_rows(metadata_csv)
    identities = sorted({row["identity"] for row in rows})
    if len(identities) < num_folds:
        raise ValueError(f"Need at least {num_folds} identities, found {len(identities)}.")

    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(identities), start=1):
        train_ids = {identities[index] for index in train_idx}
        val_ids = {identities[index] for index in val_idx}
        train_rows = [row for row in rows if row["identity"] in train_ids]
        val_rows = [row for row in rows if row["identity"] in val_ids]
        write_dict_rows(output_dir / f"fold_{fold_index:02d}_train.csv", train_rows)
        write_dict_rows(output_dir / f"fold_{fold_index:02d}_val.csv", val_rows)
        write_fold_summary(output_dir / f"fold_{fold_index:02d}_summary.json", train_rows, val_rows)
    print(f"wrote {num_folds} subject-wise folds to {output_dir}")


def augment_csv(input_csv: Path, image_root: Path, output_csv: Path, output_subdir: str) -> None:
    rows = read_metadata_rows(input_csv)
    output_rows: list[dict[str, str]] = []
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    for row in tqdm(rows, desc="paper augment"):
        image_path = Path(row["image_path"])
        source_path = image_path if image_path.is_absolute() else image_root / image_path
        image = Image.open(source_path).convert("L")

        for angle in rotations:
            for flip in [False, True]:
                augmented = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                suffix = f"rot_{angle:+d}".replace("+", "p").replace("-", "m")
                if flip:
                    augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    suffix += "_flip"

                relative_source = relative_image_path(row["image_path"], image_root)
                relative_no_aligned = Path(*relative_source.parts[1:]) if relative_source.parts[:1] == ("aligned",) else relative_source
                target_rel = Path(output_subdir) / relative_no_aligned.parent / f"{relative_no_aligned.stem}_{suffix}.png"
                target_path = image_root / target_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                augmented.save(target_path)

                output_rows.append(
                    {
                        "image_path": str(target_rel),
                        "expression": row["expression"],
                        "identity": row["identity"],
                    }
                )

    write_dict_rows(output_csv, output_rows)
    write_metadata_report(output_csv.with_suffix(".summary.json"), output_rows)
    print(f"wrote {len(output_rows)} augmented rows to {output_csv}")


def validate_csv(metadata_csv: Path, image_root: Path) -> None:
    rows = read_metadata_rows(metadata_csv)
    missing = []
    by_expression_identity: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path = Path(row["image_path"])
        full_path = image_path if image_path.is_absolute() else image_root / image_path
        if not full_path.exists():
            missing.append(str(full_path))
        by_expression_identity[row["expression"]].add(row["identity"])

    unpairable = {
        expression: sorted(identities)
        for expression, identities in by_expression_identity.items()
        if len(identities) < 2
    }
    report = build_report(rows)
    report["missing_paths"] = len(missing)
    report["unpairable_expressions"] = unpairable
    print(json.dumps(report, indent=2))
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths are missing. First missing path: {missing[0]}")
    if unpairable:
        raise ValueError(f"Some expressions cannot form different-identity pairs: {sorted(unpairable)}")


def build_output_relative_path(sample: RawSample) -> Path:
    safe_identity = sanitize(sample.identity)
    safe_expression = sanitize(sample.expression)
    safe_stem = sanitize(sample.image_path.stem)
    return Path(safe_expression) / safe_identity / f"{safe_stem}.png"


def write_metadata_csv(path: Path, samples: Iterable[ProcessedSample]) -> None:
    rows = [
        {
            "image_path": sample.image_path,
            "expression": sample.expression,
            "identity": sample.identity,
        }
        for sample in samples
    ]
    write_dict_rows(path, rows)
    write_metadata_report(path.with_suffix(".summary.json"), rows)


def merge_split_metadata(metadata_dir: Path) -> None:
    merged: list[dict[str, str]] = []
    for split_csv in sorted(metadata_dir.glob("*.csv")):
        if split_csv.name == "all.csv" or split_csv.name.startswith("fold_"):
            continue
        merged.extend(read_metadata_rows(split_csv))
    write_dict_rows(metadata_dir / "all.csv", merged)
    write_metadata_report(metadata_dir / "all.summary.json", merged)


def write_dict_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "expression", "identity"])
        writer.writeheader()
        writer.writerows(rows)


def read_metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def relative_image_path(path_text: str, image_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(image_root)
    except ValueError:
        return Path(path.name)


def whiten_grayscale_image(image: Image.Image) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    standardized = (values - values.mean()) / (values.std() + 1e-6)
    standardized = np.clip(standardized, -3.0, 3.0)
    normalized = ((standardized + 3.0) / 6.0 * 255.0).astype(np.uint8)
    return Image.fromarray(normalized, mode="L")


def write_summary_json(path: Path, samples: list[ProcessedSample], failed: list[str], attempted: int) -> None:
    rows = [
        {"image_path": sample.image_path, "expression": sample.expression, "identity": sample.identity}
        for sample in samples
    ]
    report = build_report(rows)
    report["attempted_raw_samples"] = attempted
    report["aligned_samples"] = len(samples)
    report["failed_alignment"] = len(failed)
    path.write_text(json.dumps(report, indent=2))


def write_metadata_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(build_report(rows), indent=2))


def write_fold_summary(path: Path, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> None:
    train_ids = {row["identity"] for row in train_rows}
    val_ids = {row["identity"] for row in val_rows}
    report = {
        "train": build_report(train_rows),
        "val": build_report(val_rows),
        "identity_overlap": sorted(train_ids.intersection(val_ids)),
    }
    path.write_text(json.dumps(report, indent=2))


def build_report(rows: list[dict[str, str]]) -> dict:
    expression_counts = Counter(row["expression"] for row in rows)
    identities_per_expression: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identities_per_expression[row["expression"]].add(row["identity"])
    return {
        "rows": len(rows),
        "num_classes": len(expression_counts),
        "class_counts": dict(sorted(expression_counts.items())),
        "num_identities": len({row["identity"] for row in rows}),
        "identities_per_expression": {
            expression: len(identities)
            for expression, identities in sorted(identities_per_expression.items())
        },
    }


def infer_expression_from_path(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    for part in reversed(lowered_parts):
        if part in OULU_EXPRESSIONS:
            return OULU_EXPRESSIONS[part]
    return None


def infer_identity_from_path(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"[PpSs]?\d{2,4}", part):
            return part
    match = re.search(r"([PpSs]\d{2,4})", path.as_posix())
    if match:
        return match.group(1)
    return None


def find_image(root: Path, filename: str) -> Path | None:
    candidates = [root / filename]
    stem = Path(filename).stem
    for suffix in IMAGE_EXTENSIONS:
        candidates.append(root / f"{stem}{suffix}")
        candidates.append(root / "Image" / "aligned" / f"{stem}{suffix}")
        candidates.append(root / "Image" / "original" / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{stem}.*"))
    return matches[0] if matches else None


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


if __name__ == "__main__":
    main()
