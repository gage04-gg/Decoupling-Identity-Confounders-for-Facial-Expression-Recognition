from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from .losses import (
    discriminator_loss,
    encoder_adversarial_loss,
    estimate_global_mi,
    estimate_local_mi,
)
from .model import DICEFER


@dataclass
class DICEFERConfig:
    mu_exp: float = 0.5
    nu_exp: float = 1.0
    mu_id: float = 0.5
    nu_id: float = 1.0
    delta: float = 0.1
    zeta_adv: float = 0.01
    learning_rate: float = 1e-4
    classifier_learning_rate: float = 1e-4
    encoder_lr_multiplier: float = 1.0
    label_smoothing: float = 0.0
    weight_decay: float = 0.0
    amp: bool = True

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class DICEFERTrainer:
    def __init__(self, model: DICEFER, config: DICEFERConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.expression_optimizer = torch.optim.Adam(
            list(model.expression_encoder.parameters())
            + list(model.exp_global_stats.parameters())
            + list(model.exp_local_stats.parameters()),
            lr=config.learning_rate,
        )
        self.expression_optimizer.add_param_group(
            {
                "params": list(model.exp_global_stats_n.parameters())
                + list(model.exp_local_stats_n.parameters())
            }
        )
        self.identity_optimizer = torch.optim.Adam(
            list(model.identity_encoder.parameters())
            + list(model.id_global_stats.parameters())
            + list(model.id_local_stats.parameters()),
            lr=config.learning_rate,
        )
        self.identity_optimizer.add_param_group(
            {
                "params": list(model.id_global_stats_n.parameters())
                + list(model.id_local_stats_n.parameters())
            }
        )
        self.discriminator_optimizer = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config.learning_rate,
        )
        self.classifier_optimizer = torch.optim.Adam(
            [
                {
                    "params": model.expression_encoder.parameters(),
                    "lr": config.classifier_learning_rate * config.encoder_lr_multiplier,
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": config.classifier_learning_rate,
                },
            ],
            weight_decay=config.weight_decay,
        )

    def train_expression_epoch(self, loader: Iterable, epoch: int) -> dict[str, float]:

        self.model.train()
        metrics = MetricTracker()
        if hasattr(getattr(loader, "dataset", None), "set_epoch"):
            loader.dataset.set_epoch(epoch)

        for batch in tqdm(loader, desc=f"expression epoch {epoch}", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)
            exp_m = self.model.encode_expression(image_m)
            exp_n = self.model.encode_expression(image_n)
            mi = self.config.mu_exp * (
                estimate_global_mi(self.model.exp_global_stats, exp_m.global_features, exp_n.embedding)
                + estimate_global_mi(self.model.exp_global_stats_n, exp_n.global_features, exp_m.embedding)
            ) + self.config.nu_exp * (
                estimate_local_mi(self.model.exp_local_stats, exp_m.local_features, exp_n.embedding)
                + estimate_local_mi(self.model.exp_local_stats_n, exp_n.local_features, exp_m.embedding)
            )
            l1 = F.l1_loss(exp_m.embedding, exp_n.embedding)
            loss = -mi + self.config.delta * l1
            self.expression_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.expression_encoder.parameters())
                + list(self.model.exp_global_stats.parameters())
                + list(self.model.exp_local_stats.parameters())
                + list(self.model.exp_global_stats_n.parameters())
                + list(self.model.exp_local_stats_n.parameters()),
                max_norm=10.0,
            )
            self.expression_optimizer.step()

            metrics.update(loss=loss, expression_mi=mi, expression_l1=l1)
        return metrics.compute()

    def train_identity_epoch(self, loader: Iterable, epoch: int) -> dict[str, float]:
        
        self.model.train()
        self.model.expression_encoder.eval()
        for param in self.model.expression_encoder.parameters():
            param.requires_grad_(False)

        metrics = MetricTracker()
        if hasattr(getattr(loader, "dataset", None), "set_epoch"):
            loader.dataset.set_epoch(epoch)

        for batch in tqdm(loader, desc=f"identity epoch {epoch}", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)
            with torch.no_grad():
                e_m = self.model.encode_expression(image_m).embedding
                e_n = self.model.encode_expression(image_n).embedding
            i_m_det = self.model.encode_identity(image_m).embedding.detach()
            i_n_det = self.model.encode_identity(image_n).embedding.detach()

            disc_loss = (
                discriminator_loss(self.model.discriminator, e_m, i_m_det)
                + discriminator_loss(self.model.discriminator, e_n, i_n_det)
            )
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            self.discriminator_optimizer.step()
            for parameter in self.model.discriminator.parameters():
                parameter.requires_grad_(False)
            id_m = self.model.encode_identity(image_m)
            id_n = self.model.encode_identity(image_n)
            t_m = torch.cat([e_m, id_m.embedding], dim=1)   
            t_n = torch.cat([e_n, id_n.embedding], dim=1)
            id_mi = self.config.mu_id * (
                estimate_global_mi(self.model.id_global_stats, id_m.global_features, t_m)
                + estimate_global_mi(self.model.id_global_stats_n, id_n.global_features, t_n)
            ) + self.config.nu_id * (
                estimate_local_mi(self.model.id_local_stats, id_m.local_features, t_m)
                + estimate_local_mi(self.model.id_local_stats_n, id_n.local_features, t_n)
            )
            adv = (
                encoder_adversarial_loss(self.model.discriminator, e_m, id_m.embedding)
                + encoder_adversarial_loss(self.model.discriminator, e_n, id_n.embedding)
            )
            loss = -id_mi + self.config.zeta_adv * adv

            self.identity_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.identity_encoder.parameters())
                + list(self.model.id_global_stats.parameters())
                + list(self.model.id_local_stats.parameters())
                + list(self.model.id_global_stats_n.parameters())
                + list(self.model.id_local_stats_n.parameters()),
                max_norm=10.0,
            )
            self.identity_optimizer.step()
            for parameter in self.model.discriminator.parameters():
                parameter.requires_grad_(True)

            metrics.update(loss=loss, identity_mi=id_mi, adversarial=adv, discriminator=disc_loss)
        return metrics.compute()
    def train_classifier_epoch(
        self,
        loader: Iterable,
        epoch: int,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.train()
        for parameter in self.model.expression_encoder.parameters():
            parameter.requires_grad_(True)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        metrics = MetricTracker()
        for batch in tqdm(
            loader,
            desc=f"classifier epoch {epoch}",
            leave=False,
        ):
            image, target = self._unpack_classification_batch(batch)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                expression = self.model.encode_expression(image).embedding
                logits = self.model.classifier(expression)
                loss = F.cross_entropy(
                    logits,
                    target,
                    weight=class_weights,
                    label_smoothing=self.config.label_smoothing,
                )

            self.classifier_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.classifier_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.expression_encoder.parameters())
                + list(self.model.classifier.parameters()),
                max_norm=5.0,
            )
            self.scaler.step(self.classifier_optimizer)
            self.scaler.update()

            accuracy = (
                logits.argmax(dim=1) == target
            ).float().mean()

            metrics.update(
                loss=loss,
                accuracy=accuracy,
            )
        return metrics.compute()

    @torch.no_grad()
    def evaluate_classifier(
        self,
        loader: Iterable,
        class_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        prefix: str = "evaluation",
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in tqdm(loader, desc="evaluate", leave=False):
            image, target = self._unpack_classification_batch(batch)
            logits = self.model.classify_expression(image)
            loss = F.cross_entropy(logits, target)
            predictions = logits.argmax(dim=1)

            batch_size = target.numel()
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            y_true.extend(target.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

        if total_items == 0:
            return {}

        labels = list(range(len(class_names))) if class_names is not None else sorted(set(y_true) | set(y_pred))
        target_names = class_names if class_names is not None else [str(label) for label in labels]
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
        metrics = {
            "loss": total_loss / total_items,
            "accuracy": accuracy,
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
        }

        if output_dir is not None:
            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            self._save_evaluation_outputs(Path(output_dir), prefix, metrics, report, matrix, target_names)

        return metrics

    @torch.no_grad()
    @torch.no_grad()
    def estimate_mig(self, loader: Iterable) -> float:
        self.model.eval()
        expression_mi_vals = []
        identity_mi_vals = []
        for batch in tqdm(loader, desc="estimate MIG", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)
            exp_m = self.model.encode_expression(image_m)
            exp_n = self.model.encode_expression(image_n)
            id_m = self.model.encode_identity(image_m)
            expression_mi = estimate_global_mi(
                self.model.exp_global_stats,
                exp_m.global_features,
                exp_n.embedding,
            )
            identity_mi = estimate_global_mi(
                self.model.exp_global_stats,
                id_m.global_features,
                exp_n.embedding,
            )

            expression_mi_vals.append(expression_mi.detach())
            identity_mi_vals.append(identity_mi.detach())

        expression_mi = torch.stack(expression_mi_vals).mean().item()
        identity_mi = torch.stack(identity_mi_vals).mean().item()

        mig = expression_mi - identity_mi

        return float(mig)

    def save_checkpoint(self, path: str | Path, classes: list[str], epoch: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "classes": classes,
                "model": self.model.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_m = batch["image_m"].to(self.device, non_blocking=True)
        image_n = batch["image_n"].to(self.device, non_blocking=True)
        target = batch["expression"]
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.long)
        return image_m, image_n, target.to(self.device, non_blocking=True).long()

    def _unpack_classification_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image = batch["image"] if "image" in batch else batch["image_m"]
        target = batch["expression"]
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.long)
        return (
            image.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True).long(),
        )

    def _save_evaluation_outputs(
        self,
        output_dir: Path,
        prefix: str,
        metrics: dict[str, float],
        report: dict,
        matrix: np.ndarray,
        class_names: list[str],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
        write_flat_metrics_csv(output_dir / f"{prefix}_metrics.csv", metrics)
        (output_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2))
        write_classification_report_csv(output_dir / f"{prefix}_classification_report.csv", report)
        write_confusion_matrix_csv(output_dir / f"{prefix}_confusion_matrix.csv", matrix, class_names)
        plot_confusion_matrix(output_dir / f"{prefix}_confusion_matrix.png", matrix, class_names)
        write_paper_results_table(
            output_dir / f"{prefix}_paper_table",
            metrics,
            class_names,
            report,
        )


class MetricTracker:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, **items: torch.Tensor) -> None:
        self.count += 1
        for key, value in items.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value.detach().cpu())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}




def write_flat_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def write_classification_report_csv(path: Path, report: dict) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "value": values})
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        plot_confusion_matrix_with_pillow(path, matrix, class_names)
        return

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums,
                     out=np.zeros_like(matrix, dtype=float),
                     where=row_sums != 0)

    n = len(class_names)
    fig_size = max(5, 0.9 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    cap_names = [c.capitalize() for c in class_names]
    ax.set_xticklabels(cap_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cap_names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    thresh = 0.5
    for r in range(n):
        for c in range(n):
            val = norm[r, c]
            color = "white" if val > thresh else "black"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=7 if n > 6 else 9)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix_with_pillow(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    cell = 84
    label_space = 150
    width = label_space + cell * len(class_names) + 20
    height = label_space + cell * len(class_names) + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index, name in enumerate(class_names):
        x = label_space + index * cell + cell // 2
        draw.text((x - 18, 18), name[:14], fill="black", font=font)
        y = label_space + index * cell + cell // 2
        draw.text((12, y - 5), name[:18], fill="black", font=font)

    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = float(normalized[row, col])
            shade = int(255 - value * 190)
            x0 = label_space + col * cell
            y0 = label_space + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=(shade, shade, 255), outline=(80, 80, 120))
            text = f"{value:.3f}"
            draw.text((x0 + 18, y0 + 34), text, fill="black", font=font)

    draw.text((label_space + cell * len(class_names) // 2 - 24, height - 18), "Predicted", fill="black", font=font)
    draw.text((12, label_space - 24), "True", fill="black", font=font)
    image.save(path)

def write_paper_results_table(
    path: Path,
    metrics: dict[str, float],
    class_names: list[str],
    report: dict,
) -> None:
    accuracy_pct = round(metrics["accuracy"] * 100, 2)
    precision    = round(metrics["precision_macro"], 3)
    recall       = round(metrics["recall_macro"], 3)
    f1           = round(metrics["f1_macro"], 3)

    rows = [
        {"Method": "DICE-FER (ours)",
         "Setting": "image-based",
         "Expression": str(len(class_names)),
         "Accuracy (%)": accuracy_pct,
         "Precision": precision,
         "Recall": recall,
         "F1 Score": f1},
    ]
    with path.with_suffix(".csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 1.6))
        ax.axis("off")
        col_labels = ["Method", "Setting", "Expressions", "Accuracy (%)", "Precision", "Recall", "F1 Score"]
        cell_data  = [["DICE-FER (ours)", "image-based", str(len(class_names)),
                        f"{accuracy_pct:.2f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]]
        tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)
        # Bold header
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#EEEDFE")
                cell.set_text_props(weight="bold")
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass   
