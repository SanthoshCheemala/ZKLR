#!/usr/bin/env python3
import csv
import math
import random
from pathlib import Path

NUM_TRAIN = 5000
NUM_TEST  = 1000
SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "synthetic_5f_train.csv"
TEST_PATH    = DATA_DIR / "synthetic_5f_test.csv"
MODEL_PATH   = DATA_DIR / "synthetic_5f_weights.txt"

def generate_dataset(n: int, rng: random.Random) -> list[dict]:
    records = []
    for _ in range(n):
        f = [rng.randint(0, 100) for _ in range(5)]
        total = sum(f)
        
        z = (total - 250) / 30.0
        p = 1.0 / (1.0 + math.exp(-z))
        label = 1 if rng.random() < p else 0
        
        records.append({"f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3], "f5": f[4], "label": label})
    return records

def save_csv(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["f1", "f2", "f3", "f4", "f5", "label"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records -> {path}")

def sigmoid(z: float) -> float:
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def train_lr(records: list[dict], lr=0.5, epochs=2000):
    X = [[r[f"f{i}"]/100.0 for i in range(1, 6)] for r in records]
    Y = [r["label"] for r in records]
    n = len(X)
    
    w = [0.0] * 5
    b = 0.0
    
    for epoch in range(epochs):
        dw = [0.0] * 5
        db = 0.0
        loss = 0.0
        for xi, yi in zip(X, Y):
            z = sum(w[j] * xi[j] for j in range(5)) + b
            pred = sigmoid(z)
            err = pred - yi
            for j in range(5):
                dw[j] += err * xi[j]
            db += err
            eps = 1e-15
            loss -= yi * math.log(pred + eps) + (1 - yi) * math.log(1 - pred + eps)
            
        for j in range(5):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
        loss /= n
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: loss={loss:.6f}")
            
    w_raw = [wj / 100.0 for wj in w]
    print(f"w_raw = {w_raw}, b_raw = {b}")
    return w_raw, b

def evaluate(records: list[dict], w: list[float], b: float) -> float:
    correct = 0
    for r in records:
        z = sum(w[j-1] * r[f"f{j}"] for j in range(1, 6)) + b
        pred = 1 if sigmoid(z) >= 0.5 else 0
        if pred == r["label"]:
            correct += 1
    return correct / len(records)

def main():
    rng = random.Random(SEED)
    print("Generating datasets...")
    train_data = generate_dataset(NUM_TRAIN, rng)
    test_data = generate_dataset(NUM_TEST, rng)
    save_csv(train_data, DATASET_PATH)
    save_csv(test_data, TEST_PATH)
    
    print("Training model...")
    w, b = train_lr(train_data)
    
    print("Evaluating...")
    print(f"Train Acc: {evaluate(train_data, w, b):.2%}")
    print(f"Test Acc: {evaluate(test_data, w, b):.2%}")
    
    print("Exporting...")
    with open(MODEL_PATH, "w") as f:
        f.write(f"Weights: {','.join([str(wj) for wj in w])}\n")
        f.write(f"Bias: {b}\n")
    print(f"Saved weights -> {MODEL_PATH}")

if __name__ == '__main__':
    main()
