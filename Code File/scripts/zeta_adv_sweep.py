from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_ZETAS = [0.0, 0.005, 0.010, 0.025, 0.04, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the paper's Fig. 4 zeta_adv ablation.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/zeta_adv_sweep")
    parser.add_argument("--pretrained-resnet", default=None, help="Path to CASIA-WebFace ResNet-18 weights.")
    parser.add_argument("--zetas", nargs="+", type=float, default=DEFAULT_ZETAS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--runtime-augment", choices=["paper-random", "none"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for zeta in args.zetas:
        run_dir = output_dir / f"zeta_{format_zeta(zeta)}"
        command = [
            sys.executable,
            "train.py",
            "--train-csv",
            args.train_csv,
            "--val-csv",
            args.val_csv,
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--classifier-epochs",
            str(args.classifier_epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--zeta-adv",
            str(zeta),
            "--input-channels",
            str(args.input_channels),
            "--runtime-augment",
            args.runtime_augment,
            "--seed",
            str(args.seed),
        ]
        if args.image_root is not None:
            command.extend(["--image-root", args.image_root])
        if args.pretrained_resnet is not None:
            command.extend(["--pretrained-resnet", args.pretrained_resnet])

        subprocess.run(command, check=True)
        metrics_path = run_dir / "validation_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        rows.append({"zeta_adv": zeta, **metrics})
        write_json(output_dir / "zeta_adv_sweep.json", rows)
        write_csv(output_dir / "zeta_adv_sweep.csv", rows)
        plot_curve(output_dir / "zeta_adv_sweep.png", rows)


def format_zeta(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def write_json(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "")) for field in fieldnames))
    path.write_text("\n".join(lines) + "\n")


def plot_curve(path: Path, rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_curve_with_pillow(path, rows)
        return
    from collections import defaultdict
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get("dataset", "CK+")
        by_dataset[key].append(row)
    styles = {
        "CK+":        dict(marker="o", linestyle="-",  color="#E63946", label="CK+"),
        "Oulu-CASIA": dict(marker="x", linestyle="--", color="#2A9D8F", label="Oulu-CASIA"),
        "RAF-DB":     dict(marker="s", linestyle="-.", color="#457B9D", label="RAF-DB"),
        "AffectNet":  dict(marker="^", linestyle=":",  color="#6A0572", label="AffectNet"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset_name, dataset_rows in sorted(by_dataset.items()):
        dataset_rows = sorted(dataset_rows, key=lambda r: r["zeta_adv"])
        xs = [r["zeta_adv"] for r in dataset_rows]
        ys = [r["accuracy"] * 100.0 for r in dataset_rows]
        style = styles.get(dataset_name, dict(marker="o", linestyle="-", label=dataset_name))
        ax.plot(xs, ys, **style, linewidth=1.8, markersize=6)

    ax.set_xlabel(r"$\zeta^{\mathrm{adv}}$", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(r"Effect of $\zeta^{\mathrm{adv}}$ on expression representation", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 0.005, 0.010, 0.025, 0.04, 0.05])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curve_with_pillow(path: Path, rows: list[dict[str, float]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = sorted(rows, key=lambda row: row["zeta_adv"])
    width, height = 900, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 40, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [row["zeta_adv"] for row in rows]
    y_values = [row["accuracy"] * 100.0 for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min = min(y_values) if y_values else 0.0
    y_max = max(y_values) if y_values else 1.0
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def to_xy(x_value: float, y_value: float) -> tuple[int, int]:
        x = margin_left + int((x_value - x_min) / (x_max - x_min) * plot_width)
        y = margin_top + plot_height - int((y_value - y_min) / (y_max - y_min) * plot_height)
        return x, y

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height), fill="black", width=2)
    points = [to_xy(x, y) for x, y in zip(x_values, y_values)]
    if len(points) > 1:
        draw.line(points, fill=(30, 90, 180), width=3)
    for point, x_value, y_value in zip(points, x_values, y_values):
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(30, 90, 180))
        draw.text((x - 24, y - 22), f"{y_value:.2f}", fill="black", font=font)
        draw.text((x - 18, margin_top + plot_height + 12), f"{x_value:g}", fill="black", font=font)
    draw.text((width // 2 - 95, 12), "zeta_adv ablation", fill="black", font=font)
    draw.text((width // 2 - 34, height - 34), "zeta_adv", fill="black", font=font)
    draw.text((10, margin_top + plot_height // 2), "Accuracy (%)", fill="black", font=font)
    image.save(path)


if __name__ == "__main__":
    main()
