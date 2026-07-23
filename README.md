# DICE-FER Paper Implementation

This project implements the core method from the paper:
**Decoupling Identity Confounders for Enhanced Facial Expression Recognition:
An Information-Theoretic Approach**.

Implemented pieces:

- paired image sampling for same-expression, different-identity pairs
- ResNet-18 expression and identity encoders
- MINE/Donsker-Varadhan global and local mutual-information objectives
- expression representation swapping across paired images
- L1 expression-consistency loss
- adversarial disentanglement between expression and identity embeddings
- expression classifier trained on the learned expression space
- paper-style evaluation artifacts: metrics JSON/CSV, per-class report, and confusion matrix plot

## Pair DataLoader

Each training item returns two face images:

- same expression label
- different identity label

## CSV Metadata

```python
from src.fer_pair_dataloader import FERPairDataset, create_pair_dataloader

dataset = FERPairDataset.from_csv(
    "metadata.csv",
    image_root="images",
    path_col="image_path",
    expression_col="expression",
    identity_col="identity",
    seed=42,
    transform=train_transform,
)

loader = create_pair_dataloader(dataset, batch_size=32, shuffle=True)
```

The CSV should contain one row per image:

```csv
image_path,expression,identity
S001_happy_001.jpg,happy,S001
S002_happy_001.jpg,happy,S002
S003_sad_001.jpg,sad,S003
```

## Folder Metadata

For `root/expression/image.jpg`, identity is inferred from the filename prefix:

```python
dataset = FERPairDataset.from_image_folder(
    "images",
    identity_from="filename_prefix",
    seed=42,
)
```

For `root/expression/identity/image.jpg`, use the parent folder:

```python
dataset = FERPairDataset.from_image_folder(
    "images",
    identity_from="parent",
    seed=42,
)
```

If your dataset uses a different naming scheme, pass a callable as
`identity_from`.

## Train

Install dependencies in conda base:

```bash
conda activate base
python -m pip install -r requirements.txt
```

## Prepare Datasets

The preparation script aligns faces with MTCNN, writes 112x112 grayscale PNGs,
and creates `image_path,expression,identity` CSV metadata.

### CK+

Expected raw folders:

```text
data_raw/CK+/
  cohn-kanade-images/S005/001/*.png
  Emotion/S005/001/*.txt
```

Prepare aligned images and `metadata/all.csv`:

```bash
python scripts/prepare_datasets.py ckplus \
  --images-root "data_raw/CK+/cohn-kanade-images" \
  --emotion-root "data_raw/CK+/Emotion" \
  --output-root "data_processed/CK+" \
  --device cpu
```

Create subject-wise 10 folds:

```bash
python scripts/prepare_datasets.py make-folds \
  --metadata-csv "data_processed/CK+/metadata/all.csv" \
  --output-dir "data_processed/CK+/metadata" \
  --num-folds 10
```

Create the paper's 14 deterministic training augmentations for a fold:

```bash
python scripts/prepare_datasets.py augment-csv \
  --input-csv "data_processed/CK+/metadata/fold_01_train.csv" \
  --image-root "data_processed/CK+" \
  --output-csv "data_processed/CK+/metadata/fold_01_train_paper_aug.csv"
```

### Oulu-CASIA

Prepare aligned images from the raw Oulu folder. The script searches for
expression folder names such as `Anger`, `Disgust`, `Fear`, `Happy`,
`Sadness`, and `Surprise`, and keeps the last three frames per sequence.

```bash
python scripts/prepare_datasets.py oulu \
  --raw-root "data_raw/Oulu-CASIA" \
  --output-root "data_processed/Oulu-CASIA" \
  --device cpu
```

Create subject-wise 10 folds:

```bash
python scripts/prepare_datasets.py make-folds \
  --metadata-csv "data_processed/Oulu-CASIA/metadata/all.csv" \
  --output-dir "data_processed/Oulu-CASIA/metadata" \
  --num-folds 10
```

Create paper augmentations for each training fold:

```bash
python scripts/prepare_datasets.py augment-csv \
  --input-csv "data_processed/Oulu-CASIA/metadata/fold_01_train.csv" \
  --image-root "data_processed/Oulu-CASIA" \
  --output-csv "data_processed/Oulu-CASIA/metadata/fold_01_train_paper_aug.csv"
```

### RAF-DB

Use only the official basic-expression labels. Point `--labels-file` to
`list_patition_label.txt`.

```bash
python scripts/prepare_datasets.py rafdb \
  --images-root "data_raw/RAF-DB" \
  --labels-file "data_raw/RAF-DB/EmoLabel/list_patition_label.txt" \
  --output-root "data_processed/RAF-DB" \
  --device cpu
```

This writes:

```text
data_processed/RAF-DB/metadata/train.csv
data_processed/RAF-DB/metadata/val.csv
```

Create paper augmentations for RAF-DB training:

```bash
python scripts/prepare_datasets.py augment-csv \
  --input-csv "data_processed/RAF-DB/metadata/train.csv" \
  --image-root "data_processed/RAF-DB" \
  --output-csv "data_processed/RAF-DB/metadata/train_paper_aug.csv"
```

### AffectNet

Run once for training annotations and once for validation annotations. The
default columns are `subDirectory_filePath` and `expression`.

```bash
python scripts/prepare_datasets.py affectnet \
  --images-root "data_raw/AffectNet" \
  --annotations "data_raw/AffectNet/training.csv" \
  --output-root "data_processed/AffectNet" \
  --split train \
  --device cpu

python scripts/prepare_datasets.py affectnet \
  --images-root "data_raw/AffectNet" \
  --annotations "data_raw/AffectNet/validation.csv" \
  --output-root "data_processed/AffectNet" \
  --split val \
  --device cpu
```

Create paper augmentations for AffectNet training:

```bash
python scripts/prepare_datasets.py augment-csv \
  --input-csv "data_processed/AffectNet/metadata/train.csv" \
  --image-root "data_processed/AffectNet" \
  --output-csv "data_processed/AffectNet/metadata/train_paper_aug.csv"
```

Validate any CSV before training:

```bash
python scripts/prepare_datasets.py validate-csv \
  --metadata-csv "data_processed/CK+/metadata/fold_01_train_paper_aug.csv" \
  --image-root "data_processed/CK+"
```

### Generic CSV

If a raw dataset layout does not match the built-in parser, create a raw CSV:

```csv
image_path,expression,identity,split
raw/S001_001.png,happy,S001,all
raw/S002_001.png,happy,S002,all
```

Then align it:

```bash
python scripts/prepare_datasets.py align-csv \
  --input-csv "metadata/raw_dataset.csv" \
  --image-root "data_raw" \
  --output-root "data_processed/custom" \
  --device cpu
```

Prepare a CSV:

```csv
image_path,expression,identity
S001_happy_001.jpg,happy,S001
S002_happy_001.jpg,happy,S002
S003_sad_001.jpg,sad,S003
```

Run all three stages:

```bash
python train.py \
  --train-csv data_processed/CK+/metadata/fold_01_train_paper_aug.csv \
  --val-csv data_processed/CK+/metadata/fold_01_val.csv \
  --image-root data_processed/CK+ \
  --output-dir outputs/dicefer \
  --epochs 100 \
  --classifier-epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --zeta-adv 0.01 \
  --input-channels 1 \
  --runtime-augment none \
  --pretrained-resnet /path/to/casia-webface-resnet18.pth
```

`--pretrained-resnet` must point to ResNet-18 weights pre-trained on
CASIA-WebFace, matching Section 5.3 of the paper. The training code does not
use torchvision ImageNet weights for this flag.

`--runtime-augment` defaults to `none`, because the closest paper pipeline is
to first run `prepare_datasets.py augment-csv` and then train from the
`*_paper_aug.csv` file. Use `--runtime-augment paper-random` only when training
from an unaugmented CSV.

When a validation CSV is provided, final evaluation artifacts are written under
the output directory:

```text
validation_metrics.json
validation_metrics.csv
validation_classification_report.json
validation_classification_report.csv
validation_confusion_matrix.csv
validation_confusion_matrix.png
modified_mig.json
```

Run a later stage from a saved checkpoint:

```bash
python train.py \
  --train-csv metadata/train.csv \
  --image-root images \
  --stage identity \
  --checkpoint outputs/dicefer/expression_latest.pt
```

The paper uses aligned 112x112 grayscale face crops. This code resizes to
112x112 and applies rotation/flip augmentation, but you should do face detection
and alignment before creating the CSV for the closest reproduction.

## zeta_adv Ablation

Reproduce Figure 4 with the paper's sweep values:

```bash
python scripts/zeta_adv_sweep.py \
  --train-csv data_processed/CK+/metadata/fold_01_train_paper_aug.csv \
  --val-csv data_processed/CK+/metadata/fold_01_val.csv \
  --image-root data_processed/CK+ \
  --pretrained-resnet /path/to/casia-webface-resnet18.pth \
  --output-dir outputs/zeta_adv_sweep
```

The sweep writes `zeta_adv_sweep.json`, `zeta_adv_sweep.csv`, and
`zeta_adv_sweep.png`.
