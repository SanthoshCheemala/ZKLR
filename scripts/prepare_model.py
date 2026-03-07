#!/usr/bin/env python3
"""
Dataset Generator + Logistic Regression Trainer for ZK-LR Project (v2)

Generates a student marks dataset (integer marks → pass/fail),
trains a logistic regression model, and exports the model weights
as pre-scaled integers ready for use in the Go ZK circuit.

Usage:
    python scripts/prepare_model.py
"""

import csv
import math
import os
import random
from pathlib import Path


# ─── Configuration ──────────────────────────────────────────────
NUM_TRAIN = 3000        # training samples
NUM_TEST  = 200         # held-out test samples
PASS_THRESHOLD = 50     # marks >= 50 → likely pass
NOISE_ZONE = 10         # fuzzy zone around threshold
SEED = 42
PRECISION = 32          # 2^32 scaling for ZK circuit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "student_dataset.csv"
TEST_PATH    = DATA_DIR / "student_dataset_test.csv"
MODEL_PATH   = DATA_DIR / "model_weights.txt"


# ─── 1. Dataset Generation ─────────────────────────────────────
def generate_dataset(n: int, rng: random.Random) -> list[dict]:
    """Generate student records: marks (int 0-100), failed (0 or 1)."""
    records = []
    for _ in range(n):
        marks = rng.randint(0, 100)

        # Deterministic outside noise zone, probabilistic inside
        if marks >= PASS_THRESHOLD + NOISE_ZONE:
            failed = 0  # clearly passing
        elif marks < PASS_THRESHOLD - NOISE_ZONE:
            failed = 1  # clearly failing
        else:
            # Smooth probability in the fuzzy zone
            # P(pass) increases linearly from 0.2 to 0.8 across the zone
            dist = (marks - (PASS_THRESHOLD - NOISE_ZONE)) / (2 * NOISE_ZONE)
            p_pass = 0.2 + 0.6 * dist
            failed = 0 if rng.random() < p_pass else 1

        records.append({"marks": marks, "failed": failed})
    return records


def save_csv(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["marks", "failed"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} records → {path}")


# ─── 2. Logistic Regression (from scratch, no sklearn) ─────────
def sigmoid(z: float) -> float:
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train_lr(records: list[dict], lr=0.5, epochs=5000) -> tuple[float, float]:
    """
    Train logistic regression: P(pass) = sigmoid(w * marks + b)
    where pass means failed=0.

    Features are normalized (marks/100) internally for stable gradients.
    Returned w, b are un-normalized to work with raw integer marks.
    """
    # Extract features and labels: y=1 means PASS (failed=0)
    # Normalize marks to [0, 1] for stable training
    X = [r["marks"] / 100.0 for r in records]
    Y = [1 - r["failed"] for r in records]  # 1=pass, 0=fail
    n = len(X)

    w, b = 0.0, 0.0
    best_loss = float("inf")

    for epoch in range(epochs):
        dw, db = 0.0, 0.0
        loss = 0.0
        for xi, yi in zip(X, Y):
            z = w * xi + b
            pred = sigmoid(z)
            err = pred - yi
            dw += err * xi
            db += err
            # cross-entropy loss
            eps = 1e-15
            loss -= yi * math.log(pred + eps) + (1 - yi) * math.log(1 - pred + eps)

        w -= lr * dw / n
        b -= lr * db / n
        loss /= n

        if loss < best_loss:
            best_loss = loss

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:4d}: loss={loss:.6f}, w={w:.8f}, b={b:.8f}")

    # Un-normalize: original z = w_norm * (marks/100) + b
    #             = (w_norm/100) * marks + b
    w_raw = w / 100.0
    b_raw = b
    print(f"\n  Normalized:   w={w:.10f}, b={b:.10f}")
    print(f"  Un-normalized: w={w_raw:.10f}, b={b_raw:.10f}")

    return w_raw, b_raw


def evaluate(records: list[dict], w: float, b: float) -> float:
    """Evaluate accuracy on dataset."""
    correct = 0
    for r in records:
        z = w * r["marks"] + b
        pred_pass = sigmoid(z) >= 0.5
        actual_pass = r["failed"] == 0
        if pred_pass == actual_pass:
            correct += 1
    return correct / len(records)


# ─── 3. Export Model Weights ───────────────────────────────────
def export_weights(w: float, b: float, path: Path):
    """Save model weights as both float and pre-scaled integers."""
    scale = 2 ** PRECISION
    w_scaled = round(w * scale)
    b_scaled = round(b * scale)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# ZK-LR Model Weights (v2)\n")
        f.write(f"# Trained on {NUM_TRAIN} samples, threshold ~{PASS_THRESHOLD}\n\n")
        f.write(f"# Float values\n")
        f.write(f"w = {w:.10f}\n")
        f.write(f"b = {b:.10f}\n\n")
        f.write(f"# Pre-scaled for Go ZK circuit (× 2^{PRECISION})\n")
        f.write(f"W_SCALED = {w_scaled}\n")
        f.write(f"B_SCALED = {b_scaled}\n")

    print(f"  Saved weights → {path}")
    return w_scaled, b_scaled


# ─── Main ──────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)

    print("=" * 60)
    print("  ZK-LR Model Preparation (v2)")
    print("=" * 60)

    # 1. Generate datasets
    print("\n[1] Generating datasets...")
    train_data = generate_dataset(NUM_TRAIN, rng)
    test_data  = generate_dataset(NUM_TEST, rng)
    save_csv(train_data, DATASET_PATH)
    save_csv(test_data, TEST_PATH)

    # Show distribution
    train_pass = sum(1 for r in train_data if r["failed"] == 0)
    print(f"  Train: {train_pass} pass / {NUM_TRAIN - train_pass} fail")

    # 2. Train model
    print("\n[2] Training logistic regression...")
    w, b = train_lr(train_data)
    print(f"\n  Final: w = {w:.10f}, b = {b:.10f}")

    # 3. Evaluate
    print("\n[3] Evaluating...")
    train_acc = evaluate(train_data, w, b)
    test_acc  = evaluate(test_data, w, b)
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")

    # 4. Export
    print("\n[4] Exporting model weights...")
    w_scaled, b_scaled = export_weights(w, b, MODEL_PATH)

    # 5. Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  w = {w:.10f}  →  W_SCALED = {w_scaled}")
    print(f"  b = {b:.10f}  →  B_SCALED = {b_scaled}")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")
    print()
    print("  Go constants to use:")
    print(f'    var W_SCALED = new(big.Int).SetInt64({w_scaled})')
    print(f'    var B_SCALED = new(big.Int).SetInt64({b_scaled})')
    print("=" * 60)


if __name__ == "__main__":
    main()
