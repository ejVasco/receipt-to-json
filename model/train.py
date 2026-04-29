"""
train.py
leave one out cross validiation (loocv) training for receipt MLp

for each of the x receipts:
    - train on the other x-11
    - evaluate on the held-out receipt
    - report per-class f1 and macro f1

final model is then retrained on all data and saved

usage:
    python -m model.train
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

import config
from model.dataset import (
    LABELS,
    NUM_CLASSES,
    ReceiptDataset,
    get_all_stems,
    loocv_splits,
)
from model.model import ReceiptMLP

# -- hyperparameters ---
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_fold(
    train_stems, val_stems, verbose=False
) -> tuple[list[int], list[int], ReceiptMLP] | None:
    """train and eval one LOOCV fold, returns (all_preds, all_targets)."""

    train_ds = ReceiptDataset(config.OCR_DIR, config.LABELS_DIR, stems=train_stems)
    val_ds = ReceiptDataset(config.OCR_DIR, config.LABELS_DIR, stems=val_stems)

    if len(train_ds) == 0 or len(val_ds) == 0:
        return None

    # class weights to handle imbalance (OTHER dominates)
    label_counts = np.zeros(NUM_CLASSES)
    for _, y in train_ds:
        label_counts[int(y.item())] += 1
    label_counts = np.where(label_counts == 0, 1, label_counts)
    class_weights = torch.tensor(1.0 / label_counts, dtype=torch.float32).to(DEVICE)  # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    model = ReceiptMLP(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    loss = torch.tensor(0.0)  # type: ignore[attr-defined]
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}/{EPOCHS}  loss={loss.item():.4f}")

    # eval
    model.eval()
    all_preds, all_targets = [], []
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.tolist())

    return all_preds, all_targets, model


def main():

    import os

    print("cwd:", os.getcwd())
    print("ocr exists:", config.OCR_DIR.exists())
    print("labels exists:", config.LABELS_DIR.exists())

    stems = get_all_stems(config.OCR_DIR, config.LABELS_DIR)
    print(f"Found {len(stems)} receipt pairs: {stems}\n")
    print(f"Device: {DEVICE}")
    print(f"Running LOOCV ({len(stems)} folds)...\n")

    all_preds_global = []
    all_targets_global = []

    for fold_idx, (train_stems, val_stems) in enumerate(loocv_splits(stems)):
        print(f"Fold {fold_idx + 1}/{len(stems)}  held-out: {val_stems[0]}")
        result = train_one_fold(train_stems, val_stems)
        if result is None:
            continue
        preds, targets, _ = result
        all_preds_global.extend(preds)
        all_targets_global.extend(targets)

        fold_f1 = f1_score(targets, preds, average="macro", zero_division=0)  # type: ignore[call-overload]
        print(f"  macro F1 = {fold_f1:.3f}\n")

    # ----- eggregate results ---------
    if not all_targets_global:
        print("No predictions collected — check that OCR/label files are being found.")
        return
    print("=" * 40)
    print("LOOCV Aggregate Results")
    print("=" * 40)
    print(
        classification_report(
            all_targets_global,
            all_preds_global,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,  # type: ignore[call-overload]
        )
    )

    macro_f1 = f1_score(
        all_targets_global,
        all_preds_global,
        average="macro",
        zero_division=0,  # type: ignore[call-overload]
    )
    print(f"Overall macro F1 {macro_f1:.4f}")

    # -- retrain on all data, save final model -------
    print("\nRetraining on all data...")
    final_result = train_one_fold(stems, stems, verbose=True)
    if final_result is None:
        print("No data — cannot save model.")
        return
    _, _, final_model = final_result
    save_path = Path("model/receipt_mlp.pt")
    torch.save(final_model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")


if __name__ == "__main__":
    main()
