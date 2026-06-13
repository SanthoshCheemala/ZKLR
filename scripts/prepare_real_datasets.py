#!/usr/bin/env python3
"""Prepare the three real-world LR datasets and run the fidelity study (T4).

Datasets (all binary classification, privacy-sensitive domains):
  wbc    — Breast Cancer Wisconsin (Original), UCI #15.
           9 features, raw integers 1..10 → zero input-quantization error,
           isolating the effect of weight quantization.
  pima   — Pima Indians Diabetes (NIDDK). 8 features, fixed per-feature
           scale factors to integers (documented below).
  credit — Default of Credit Card Clients (Taiwan), UCI #350. 23 features,
           min-max scaled to integers in [0, 4000] (train statistics).

For each dataset this script:
  1. downloads + caches the raw data (data/raw/),
  2. quantizes features to the circuit's 12-bit integer domain,
  3. trains float sklearn LogisticRegression (seed=42, stratified 70/30),
  4. simulates the ZK circuit's fixed-point pipeline (exact integer mirror of
     prover.ComputeWitness: 2^32 weight scale, 2^22 truncation, 2^10 sigmoid
     table index, 2^16 output, clamp at z = ±10),
  5. reports fidelity: float vs circuit accuracy, label agreement, |Δp|, AUC,
  6. writes ZKLR-format CSVs + weights files + a fidelity report.

Usage:
  python3 scripts/prepare_real_datasets.py            # all three
  python3 scripts/prepare_real_datasets.py --dataset wbc
"""

import argparse
import math
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
TEST_FRACTION = 0.3

# ─── Circuit constants (must match circuit/circuit.go + sigmoid.go) ───
PRECISION = 32                    # weights scaled by 2^32
INPUT_PRECISION = 10              # sigmoid table index resolution
OUTPUT_PRECISION = 16             # sigmoid output resolution
MODEL_OFFSET = 1000
SIGMOID_OFFSET = 10
SHIFT = 1 << (PRECISION - INPUT_PRECISION)          # 2^22
LOWER = (MODEL_OFFSET - SIGMOID_OFFSET) << INPUT_PRECISION
UPPER = (MODEL_OFFSET + SIGMOID_OFFSET) << INPUT_PRECISION
X_MAX = (1 << 12) - 1             # 12-bit feature range check

RAW_DIR = "data/raw"
DATA_DIR = "data"
RESULTS_DIR = "results"


def fetch(url: str, dest: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, dest)
    if not os.path.exists(path):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


# ─── Exact integer mirror of prover.ComputeWitness ───────────────────

def circuit_predict(w_float, b_float, x_int_rows):
    """Returns (probabilities, labels) as produced by the ZK circuit."""
    # Go: int64(w * float64(2^32)) — truncation toward zero, like Python int()
    w_scaled = [int(w * float(1 << PRECISION)) for w in w_float]
    b_scaled = int(b_float * float(1 << PRECISION))

    probs, labels = [], []
    for x in x_int_rows:
        z = sum(ws * int(xi) for ws, xi in zip(w_scaled, x)) + b_scaled
        z_shifted = z + (MODEL_OFFSET << PRECISION)
        if z_shifted < 0:
            raise ValueError(
                f"z below completeness domain (z={z / 2**PRECISION:.2f} < -{MODEL_OFFSET})")
        z_table = z_shifted // SHIFT
        z_clamped = min(max(z_table, LOWER), UPPER)
        z_index = z_clamped - LOWER
        z_f = z_index / float(1 << INPUT_PRECISION) - SIGMOID_OFFSET
        y = 1.0 / (1.0 + math.exp(-z_f))
        y_scaled = min(int(y * (1 << OUTPUT_PRECISION)), (1 << OUTPUT_PRECISION) - 1)
        probs.append(y_scaled / float(1 << OUTPUT_PRECISION))
        labels.append(1 if y_scaled >= 1 << (OUTPUT_PRECISION - 1) else 0)
    return np.array(probs), np.array(labels)


# ─── Dataset loaders → (X_int DataFrame, y Series, notes) ─────────────

def load_wbc():
    path = fetch(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        "breast-cancer-wisconsin.data")
    cols = ["id", "clump_thickness", "cell_size", "cell_shape", "adhesion",
            "epithelial_size", "bare_nuclei", "bland_chromatin",
            "normal_nucleoli", "mitoses", "class"]
    df = pd.read_csv(path, names=cols, na_values="?").dropna()
    y = (df["class"] == 4).astype(int)          # 4 = malignant
    X = df[cols[1:-1]].astype(int)               # raw integers 1..10
    return X, y, "features used raw (integers 1..10) — no input quantization"


def load_pima():
    path = fetch(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        "pima-indians-diabetes.csv")
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "pedigree", "age", "outcome"]
    df = pd.read_csv(path, names=cols)
    y = df["outcome"].astype(int)
    # Fixed per-feature scale factors to land in the 12-bit integer domain
    scales = {"pregnancies": 1, "glucose": 1, "blood_pressure": 1,
              "skin_thickness": 1, "insulin": 1, "bmi": 10,
              "pedigree": 1000, "age": 1}
    X = pd.DataFrame({c: (df[c] * s).round().astype(int) for c, s in scales.items()})
    note = "scale factors: " + ", ".join(f"{c}×{s}" for c, s in scales.items())
    return X, y, note


def load_credit():
    path = fetch(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
        "default%20of%20credit%20card%20clients.xls",
        "credit_default.xls")
    df = pd.read_excel(path, header=1)
    y = df["default payment next month"].astype(int)
    X_raw = df.drop(columns=["ID", "default payment next month"])
    return X_raw, y, "min-max scaled to integers in [0, 4000] (train statistics)"


# ─── Pipeline ─────────────────────────────────────────────────────────

def prepare(name):
    print(f"\n=== {name} ===")
    minmax = False
    if name == "wbc":
        X, y, note = load_wbc()
    elif name == "pima":
        X, y, note = load_pima()
    elif name == "credit":
        X, y, note = load_credit()
        minmax = True
    else:
        raise ValueError(name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y)

    if minmax:
        lo, hi = X_train.min(), X_train.max()
        span = (hi - lo).replace(0, 1)
        X_train = ((X_train - lo) / span * 4000).round().astype(int).clip(0, 4000)
        X_test = ((X_test - lo) / span * 4000).round().astype(int).clip(0, 4000)

    assert X_train.min().min() >= 0 and X_train.max().max() <= X_MAX, "12-bit domain violated"
    assert X_test.min().min() >= 0 and X_test.max().max() <= X_MAX, "12-bit domain violated"

    # Float reference model, trained on the integer feature representation
    clf = LogisticRegression(penalty=None, max_iter=5000, random_state=SEED)
    clf.fit(X_train, y_train)
    w, b = clf.coef_[0], float(clf.intercept_[0])

    # Completeness-domain check: |z| must stay below MODEL_OFFSET
    z_all = X_test.values @ w + b
    max_abs_z = float(np.abs(z_all).max())
    saturated = float((np.abs(z_all) > SIGMOID_OFFSET).mean() * 100)
    if max_abs_z >= MODEL_OFFSET:
        raise SystemExit(f"  ERROR: max |z| = {max_abs_z:.1f} exceeds completeness domain ±{MODEL_OFFSET}")

    # Float vs circuit predictions on the test set
    p_float = clf.predict_proba(X_test)[:, 1]
    l_float = (p_float >= 0.5).astype(int)
    p_circ, l_circ = circuit_predict(w, b, X_test.values)

    acc_float = accuracy_score(y_test, l_float)
    acc_circ = accuracy_score(y_test, l_circ)
    agreement = float((l_float == l_circ).mean() * 100)
    dp = np.abs(p_float - p_circ)
    auc_float = roc_auc_score(y_test, p_float)
    auc_circ = roc_auc_score(y_test, p_circ)

    # ─── Emit ZKLR artifacts ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    feat_cols = list(X.columns)
    for split, Xs, ys in (("train", X_train, y_train), ("test", X_test, y_test)):
        out = Xs.copy()
        out["label"] = ys.values
        out.to_csv(os.path.join(DATA_DIR, f"{name}_{split}.csv"), index=False)

    weights_path = os.path.join(DATA_DIR, f"{name}_weights.txt")
    with open(weights_path, "w") as f:
        f.write(f"# {name}: float LR weights (trained on quantized integer features)\n")
        f.write(f"# {note}\n")
        f.write(f"# train/test: {len(X_train)}/{len(X_test)}, seed={SEED}, stratified\n")
        f.write(f"# features: {','.join(feat_cols)}\n")
        f.write("Weights: " + ",".join(repr(float(x)) for x in w) + "\n")
        f.write(f"Bias: {float(b)!r}\n")
        f.write("# run:\n")
        f.write(f"# go run ./cmd/batch_predict -dataset=data/{name}_test.csv "
                f"-weights=\"{','.join(repr(float(x)) for x in w)}\" -bias={float(b)!r}\n")

    report_path = os.path.join(RESULTS_DIR, f"fidelity_{name}.txt")
    with open(report_path, "w") as f:
        f.write(f"ZKLR Fidelity Report — {name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"samples (train/test):    {len(X_train)}/{len(X_test)}\n")
        f.write(f"features:                {len(feat_cols)}\n")
        f.write(f"quantization:            {note}\n")
        f.write(f"max |z| on test:         {max_abs_z:.3f} (domain ±{MODEL_OFFSET})\n")
        f.write(f"saturated (|z|>10):      {saturated:.2f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"float accuracy:          {acc_float * 100:.2f}%\n")
        f.write(f"circuit accuracy:        {acc_circ * 100:.2f}%\n")
        f.write(f"accuracy delta:          {(acc_circ - acc_float) * 100:+.4f} pp\n")
        f.write(f"label agreement:         {agreement:.2f}%\n")
        f.write(f"max |dp|:                {dp.max():.6f}\n")
        f.write(f"mean |dp|:               {dp.mean():.6f}\n")
        f.write(f"float AUC:               {auc_float:.6f}\n")
        f.write(f"circuit AUC:             {auc_circ:.6f}\n")
        f.write(f"AUC delta:               {auc_circ - auc_float:+.6f}\n")

    # Fidelity figure (F6): float vs circuit probability + |Δp| histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax0.scatter(p_float, p_circ, s=8, alpha=0.5)
        ax0.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax0.set_xlabel("float model probability")
        ax0.set_ylabel("circuit probability")
        ax0.set_title(f"{name}: float vs circuit ({agreement:.1f}% label agreement)")
        ax0.grid(alpha=0.3)

        ax1.hist(dp, bins=40)
        ax1.axvline(2.66e-4, color="r", ls="--", lw=1,
                    label="analytic bound 2.66e-4")
        ax1.set_xlabel("|Δp|")
        ax1.set_ylabel("samples")
        ax1.set_title(f"{name}: quantization error (max {dp.max():.6f})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f"fig_fidelity_{name}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {fig_path}")
    except ImportError:
        pass

    print(f"  train/test: {len(X_train)}/{len(X_test)}  features: {len(feat_cols)}")
    print(f"  float acc: {acc_float*100:.2f}%  circuit acc: {acc_circ*100:.2f}%  "
          f"agreement: {agreement:.2f}%")
    print(f"  max|dp|: {dp.max():.6f}  AUC delta: {auc_circ - auc_float:+.6f}  "
          f"max|z|: {max_abs_z:.2f}")
    print(f"  wrote data/{name}_train.csv, data/{name}_test.csv, {weights_path}, {report_path}")
    return name, len(X_test), len(feat_cols), acc_float, acc_circ, agreement, dp.max(), auc_circ - auc_float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wbc", "pima", "credit", "all"], default="all")
    args = ap.parse_args()

    names = ["wbc", "pima", "credit"] if args.dataset == "all" else [args.dataset]
    rows = [prepare(n) for n in names]

    print("\n" + "=" * 78)
    print(f"{'dataset':<8} {'test':>6} {'feat':>5} {'float acc':>10} {'circ acc':>10} "
          f"{'agree':>8} {'max|dp|':>10} {'dAUC':>10}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:>6} {r[2]:>5} {r[3]*100:>9.2f}% {r[4]*100:>9.2f}% "
              f"{r[5]:>7.2f}% {r[6]:>10.6f} {r[7]:>+10.6f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
