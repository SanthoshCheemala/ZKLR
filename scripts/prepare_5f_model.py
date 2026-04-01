#!/usr/bin/env python3
"""
5-Feature dataset generator + logistic regression trainer for ZK-LR.
Features: age, study_hours, sleep_hours, attendance_pct, prev_score
Label: pass (1) / fail (0)
"""
import csv, math, random
from pathlib import Path

NUM_TRAIN = 5000
NUM_TEST  = 1000
SEED      = 42
PRECISION = 32   # 2^32 scaling

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "synthetic_5f_train.csv"
TEST_PATH  = DATA_DIR / "synthetic_5f_test.csv"
MODEL_PATH = DATA_DIR / "synthetic_5f_weights.txt"

FEATURES = ["age", "study_hours", "sleep_hours", "attendance_pct", "prev_score"]

def generate(n, rng):
    records = []
    for _ in range(n):
        age          = rng.randint(18, 30)          # 18-30
        study_hours  = rng.randint(0, 12)           # hours/day
        sleep_hours  = rng.randint(4, 10)           # hours/day
        attendance   = rng.randint(50, 100)         # percent
        prev_score   = rng.randint(0, 100)          # previous exam score

        # True signal: high study, high attendance, good prev score → pass
        score = (study_hours * 4) + (attendance - 50) + (prev_score * 0.5) - (age - 18) * 0.5
        z = (score - 60) / 15.0
        p = 1.0 / (1.0 + math.exp(-z))
        label = 1 if rng.random() < p else 0
        records.append({
            "age": age, "study_hours": study_hours,
            "sleep_hours": sleep_hours, "attendance_pct": attendance,
            "prev_score": prev_score, "label": label
        })
    return records

def save_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = FEATURES + ["label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"  Saved {len(records)} records → {path}")

def sigmoid(z):
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def train(records, lr=0.3, epochs=3000):
    # Normalize: divide each feature by its max so all in [0,1]
    maxes = [30, 12, 10, 100, 100]
    X = [[r[f] / maxes[i] for i, f in enumerate(FEATURES)] for r in records]
    Y = [r["label"] for r in records]
    n = len(X)
    w = [0.0] * 5
    b = 0.0
    for epoch in range(epochs):
        dw = [0.0] * 5; db = 0.0; loss = 0.0
        for xi, yi in zip(X, Y):
            z   = sum(w[j] * xi[j] for j in range(5)) + b
            p   = sigmoid(z)
            err = p - yi
            for j in range(5):
                dw[j] += err * xi[j]
            db   += err
            eps   = 1e-15
            loss -= yi * math.log(p + eps) + (1 - yi) * math.log(1 - p + eps)
        for j in range(5):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
        loss /= n
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}: loss={loss:.6f}")

    # Un-normalize back to raw feature scale
    w_raw = [w[i] / maxes[i] for i in range(5)]
    return w_raw, b

def evaluate(records, w, b):
    correct = 0
    for r in records:
        z = sum(w[i] * r[f] for i, f in enumerate(FEATURES)) + b
        if (1 if sigmoid(z) >= 0.5 else 0) == r["label"]:
            correct += 1
    return correct / len(records)

def main():
    rng = random.Random(SEED)
    print("=" * 60)
    print("  5-Feature ZK-LR Model Preparation")
    print("=" * 60)

    print("\n[1] Generating datasets...")
    train_data = generate(NUM_TRAIN, rng)
    test_data  = generate(NUM_TEST,  rng)
    save_csv(train_data, TRAIN_PATH)
    save_csv(test_data,  TEST_PATH)
    pos = sum(r["label"] for r in train_data)
    print(f"  Label balance: {pos} pass / {NUM_TRAIN - pos} fail")

    print("\n[2] Training logistic regression ({} epochs)...".format(3000))
    w, b = train(train_data)

    print("\n[3] Evaluating...")
    tr_acc = evaluate(train_data, w, b)
    te_acc = evaluate(test_data,  w, b)
    print(f"  Train accuracy: {tr_acc:.2%}")
    print(f"  Test  accuracy: {te_acc:.2%}")

    print("\n[4] Exporting weights...")
    scale = 2 ** PRECISION
    print(f"\n{'Feature':<18} {'w (float)':>14} {'W_SCALED':>18}")
    print("-" * 52)
    for i, f in enumerate(FEATURES):
        ws = round(w[i] * scale)
        print(f"  {f:<16} {w[i]:>14.10f} {ws:>18}")
    bs = round(b * scale)
    print(f"  {'bias':<16} {b:>14.10f} {bs:>18}")

    weights_str = ",".join(str(wi) for wi in w)
    with open(MODEL_PATH, "w") as f:
        f.write(f"# 5-Feature ZK-LR Model Weights\n")
        f.write(f"# Features: {', '.join(FEATURES)}\n")
        f.write(f"# Train accuracy: {tr_acc:.2%}  Test accuracy: {te_acc:.2%}\n\n")
        f.write(f"weights = {weights_str}\n")
        f.write(f"bias    = {b}\n\n")
        for i, feat in enumerate(FEATURES):
            ws = round(w[i] * scale)
            f.write(f"# {feat}: w={w[i]:.10f}  W_SCALED={ws}\n")
        bs = round(b * scale)
        f.write(f"# bias: b={b:.10f}  B_SCALED={bs}\n")
    print(f"\n  Saved → {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  CLI command to run ZKP (batch=20, workers=3):")
    print(f'  go run ./cmd/batch_predict/main.go \\')
    print(f'    -dataset=data/synthetic_5f_test.csv \\')
    print(f'    -weights="{weights_str}" \\')
    print(f'    -bias="{b}" \\')
    print(f'    -batch=20 -workers=3')
    print("=" * 60)

if __name__ == "__main__":
    main()
