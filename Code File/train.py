from __future__ import annotations
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
import torch
from torchvision import transforms
from src.dicefer import DICEFER, DICEFERConfig, DICEFERTrainer
from src.fer_pair_dataloader import (
    FERDataset,
    FERPairDataset,
    create_dataloader,
    create_pair_dataloader,
)

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"All RNG seeds fixed to {seed}")

def worker_init_fn(worker_id: int) -> None:
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)

def resolve_device(requested: str, require_gpu: bool = False) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        print(f"✓ GPU found: {props.name}")
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        return device

    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        print("✓ Apple MPS (M-series GPU) found")
        return torch.device("mps")

    if require_gpu or requested in {"cuda", "mps"}:
        raise RuntimeError(
            "\n\n❌ NO GPU FOUND.\n"
            "torch.cuda.is_available() returned False.\n\n"
            "Possible causes:\n"
            "  1. You installed the CPU-only PyTorch build.\n"
            "     Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  2. CUDA driver version is too old for your PyTorch build.\n"
            "     Fix: Update NVIDIA drivers from nvidia.com\n"
            "  3. On Lightning AI: you are on a CPU studio — switch to L4 GPU first.\n"
            "     Fix: Click the compute selector top-right → select L4\n\n"
            f"PyTorch version: {torch.__version__}\n"
            f"Built with CUDA: {torch.version.cuda}\n"
        )

    print("⚠ No GPU found — falling back to CPU. Training will be very slow.")
    return torch.device("cpu")


class LabelEncoder:
    def __init__(self, class_to_idx: dict) -> None:
        self.class_to_idx = class_to_idx

    def __call__(self, expression: str) -> int:
        return self.class_to_idx[expression]

class ConvertToRGB:
    def __call__(self, image):
        return image.convert("RGB")


def load_compatible_checkpoint(model: DICEFER, checkpoint: dict) -> None:
    state_dict = checkpoint["model"]
    result = model.load_state_dict(state_dict, strict=False)
    directional_pairs = (
        ("exp_global_stats", "exp_global_stats_n"),
        ("exp_local_stats", "exp_local_stats_n"),
        ("id_global_stats", "id_global_stats_n"),
        ("id_local_stats", "id_local_stats_n"),
    )
    for source_name, target_name in directional_pairs:
        if not any(key.startswith(f"{target_name}.") for key in state_dict):
            getattr(model, target_name).load_state_dict(
                getattr(model, source_name).state_dict()
            )

    allowed_missing = tuple(f"{target}." for _, target in directional_pairs)
    incompatible_missing = [
        key for key in result.missing_keys if not key.startswith(allowed_missing)
    ]
    if incompatible_missing or result.unexpected_keys:
        raise RuntimeError(
            "Incompatible checkpoint: "
            f"missing={incompatible_missing}, unexpected={result.unexpected_keys}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DICE-FER from the paper implementation.")
    parser.add_argument("--train-csv", required=True, help="CSV with image_path, expression, identity columns.")
    parser.add_argument("--val-csv", default=None, help="Optional validation CSV with the same columns.")
    parser.add_argument("--image-root", default=None, help="Root for relative image paths in CSV files.")
    parser.add_argument("--output-dir", default="outputs/dicefer")
    parser.add_argument("--epochs", type=int, default=5, help="Fast default; use 100 for the paper schedule.")
    parser.add_argument("--batch-size", type=int, default=128, help="Paper uses batch size 32.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Paper uses Adam lr=1e-4.")
    parser.add_argument("--zeta-adv", type=float, default=0.01, help="Adversarial coefficient zeta_adv; Fig. 4 peaks around 0.010.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Paper uses 64-D E and I.")
    parser.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument(
        "--pretrained-resnet",
        default="imagenet",
        help=(
            "CASIA-WebFace ResNet-18 checkpoint (exact paper initialization), "
            "'imagenet' (fast strong fallback), or 'none' (not recommended)."
        ),
    )
    parser.add_argument(
        "--runtime-augment",
        choices=["paper-random", "none"],
        default="paper-random",
        help=(
            "paper-random gives the paper's rotations/flips without making a 14x larger dataset. "
            "Use none with an already expanded *_paper_aug.csv."
        ),
    )
    parser.add_argument("--stage", choices=["all", "expression", "identity", "classifier"], default="all")
    parser.add_argument("--classifier-epochs", type=int, default=5)
    parser.add_argument("--identity-epochs", type=int, default=5, help="Set 0 for an accuracy-only fast run.")
    parser.add_argument(
        "--paper-schedule",
        action="store_true",
        help="Use 5 epochs for expression, identity, and classifier stages.",
    )
    parser.add_argument("--patience", type=int, default=6, help="Classifier early stopping; 0 disables it.")
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="Evaluate each epoch and select on --val-csv (use only for a true validation set, never the test set).",
    )
    parser.add_argument("--class-balance", choices=["none", "sqrt"], default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paper_schedule:
        args.epochs = 5
        args.identity_epochs = 5
        args.classifier_epochs = 5
        args.patience = 0
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    print(f"Training on: {device}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = read_classes(args.train_csv)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    class_weights = (
        calculate_class_weights(args.train_csv, class_to_idx)
        if args.class_balance == "sqrt"
        else None
    )
    (output_dir / "classes.json").write_text(json.dumps(class_to_idx, indent=2))

    train_transform = build_transform(
        train=True,
        input_channels=args.input_channels,
        runtime_augment=args.runtime_augment,
    )
    eval_transform = build_transform(train=False, input_channels=args.input_channels, runtime_augment="none")

    train_dataset = FERPairDataset.from_csv(
        args.train_csv,
        image_root=args.image_root,
        transform=train_transform,
        target_transform=LabelEncoder(class_to_idx),
        seed=args.seed,
        image_mode="L" if args.input_channels == 1 else "RGB",
    )
    pair_train_loader = create_pair_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    classifier_dataset = FERDataset.from_csv(
        args.train_csv,
        image_root=args.image_root,
        transform=train_transform,
        target_transform=LabelEncoder(class_to_idx),
        image_mode="L" if args.input_channels == 1 else "RGB",
    )
    classifier_loader = create_dataloader(
        classifier_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_pair_loader = None
    if args.val_csv:
        val_dataset = FERDataset.from_csv(
            args.val_csv,
            image_root=args.image_root,
            transform=eval_transform,
            target_transform=LabelEncoder(class_to_idx),
            image_mode="L" if args.input_channels == 1 else "RGB",
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        if args.identity_epochs > 0:
            val_pair_dataset = FERPairDataset.from_csv(
                args.val_csv,
                image_root=args.image_root,
                transform=eval_transform,
                target_transform=LabelEncoder(class_to_idx),
                seed=args.seed,
                image_mode="L" if args.input_channels == 1 else "RGB",
            )
            val_pair_loader = create_pair_dataloader(
                val_pair_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
                persistent_workers=args.num_workers > 0,
            )

    model = DICEFER(
        num_classes=len(classes),
        embedding_dim=args.embedding_dim,
        input_channels=args.input_channels,
        pretrained_resnet=(
            None
            if args.checkpoint or args.pretrained_resnet.lower() == "none"
            else args.pretrained_resnet
        ),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        load_compatible_checkpoint(model, checkpoint)

    config = DICEFERConfig(
        learning_rate=args.lr,
        classifier_learning_rate=args.lr,
        encoder_lr_multiplier=args.encoder_lr_multiplier,
        label_smoothing=args.label_smoothing,
        zeta_adv=args.zeta_adv,
        amp=not args.no_amp,
        device=str(device),
    )
    trainer = DICEFERTrainer(model, config)

    if args.stage in {"all", "expression"}:
        for epoch in range(1, args.epochs + 1):
            metrics = trainer.train_expression_epoch(pair_train_loader, epoch)
            print_metrics("expression", epoch, metrics)
            trainer.save_checkpoint(output_dir / "expression_latest.pt", classes, epoch)

    if args.stage in {"all", "identity"}:
        for epoch in range(1, args.identity_epochs + 1):
            metrics = trainer.train_identity_epoch(pair_train_loader, epoch)
            print_metrics("identity", epoch, metrics)
            trainer.save_checkpoint(output_dir / "identity_latest.pt", classes, epoch)

    if args.stage in {"all", "classifier"}:
        best_score = float("-inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.classifier_epochs + 1):
            metrics = trainer.train_classifier_epoch(
                classifier_loader,
                epoch,
                class_weights=class_weights,
            )
            print_metrics("classifier", epoch, metrics)
            if val_loader is None or not args.select_best:
                continue
            val_metrics = trainer.evaluate_classifier(
                val_loader,
                class_names=classes,
            )
            print_metrics("validation", epoch, val_metrics)
            score = val_metrics["accuracy"]
            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                trainer.save_checkpoint(
                    output_dir / "best_classifier.pt",
                    classes,
                    epoch,
                )
            else:
                epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print("Classifier early stopping.")
                break

        best_path = output_dir / "best_classifier.pt"
        if args.select_best and best_path.exists():
            best_checkpoint = torch.load(
                best_path,
                map_location=trainer.device,
            )
            trainer.model.load_state_dict(
                best_checkpoint["model"],
            )
        trainer.save_checkpoint(
            output_dir / "dicefer_final.pt",
            classes,
            epoch,
        )

    if val_loader is not None:
        final_metrics = trainer.evaluate_classifier(
            val_loader,
            class_names=classes,
            output_dir=output_dir,
            prefix="validation",
        )
        print_metrics("validation_final", args.epochs, final_metrics)
        if val_pair_loader is not None:
            mig = trainer.estimate_mig(val_pair_loader)
            (output_dir / "modified_mig.json").write_text(
                json.dumps(
                    {"modified_mig": mig},
                    indent=2,
                )
            )
            print(f"modified_mig={mig:.4f}")


def build_transform(train: bool, input_channels: int, runtime_augment: str):
    ops = [transforms.Resize((112, 112))]
    if input_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    else:
        ops.append(ConvertToRGB())
    if train and runtime_augment == "paper-random":
        ops.extend([PaperRandomRotation(),transforms.RandomHorizontalFlip(p=0.5)])
    ops.append(transforms.ToTensor())
    mean = [0.5] * input_channels
    std = [0.5] * input_channels
    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


class PaperRandomRotation:

    def __init__(self) -> None:
        self.angles = [-15, -10, -5, 0, 5, 10, 15]

    def __call__(self, image):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(image, angle, fill=0)

def calculate_class_weights(
    csv_path: str | Path,
    class_to_idx: dict[str, int],
) -> torch.Tensor:
    counts = Counter()

    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            expression = row["expression"].strip()
            counts[expression] += 1

    weights = torch.ones(
        len(class_to_idx),
        dtype=torch.float32,
    )

    for name, index in class_to_idx.items():
        weights[index] = (
            1.0 / max(counts[name], 1) ** 0.5
        )

    return weights / weights.mean()
def read_classes(csv_path: str | Path, expression_col: str = "expression") -> list[str]:
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if expression_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV is missing required column {expression_col!r}.")
        return sorted({row[expression_col].strip() for row in reader if row[expression_col].strip()})


def print_metrics(stage: str, epoch: int, metrics: dict[str, float]) -> None:
    metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"{stage} epoch={epoch} {metric_text}")

if __name__ == "__main__":
    main()
