#!/usr/bin/env python3
"""
Generate synthetic dataset with 4 features and 2 classes (0 or 1).
Train a logistic regression model and export weights.
"""
import csv
import math
import random
from pathlib import Path

NUM_TRAIN = 5000
NUM_TEST = 1000
SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "synthetic_4f_train.csv"
TEST_PATH = DATA_DIR / "synthetic_4f_test.csv"
MODEL_PATH = DATA_DIR / "model_weights.txt"


def generate_dataset(n: int, rng: random.Random) -> list[dict]:
    """Generate dataset with 4 features and binary class label."""
    records = []
    for _ in range(n):
        # Generate 4 features in range [0, 100]
        f = [rng.randint(0, 100) for _ in range(4)]
        total = sum(f)
        
        # Use sigmoid probability based on feature sum
        z = (total - 200) / 40.0
        p = 1.0 / (1.0 + math.exp(-z))
        label = 1 if rng.random() < p else 0
        
        records.append({
            "f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3], 
            "label": label
        })
    return records


def save_csv(records: list[dict], path: Path):
    """Save records to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["f1", "f2", "f3", "f4", "label"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records -> {path}")


def sigmoid(z: float) -> float:
    """Compute sigmoid function with numerical stability."""
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train_lr(records: list[dict], lr=0.5, epochs=2000):
    """Train logistic regression with gradient descent."""
    X = [[r[f"f{i}"]/100.0 for i in range(1, 5)] for r in records]
    Y = [r["label"] for r in records]
    n = len(X)
    
    # Initialize weights
    w = [0.0] * 4
    b = 0.0
    
    for epoch in range(epochs):
        dw = [0.0] * 4
        db = 0.0
        loss = 0.0
        
        for xi, yi in zip(X, Y):
            z = sum(w[j] * xi[j] for j in range(4)) + b
            pred = sigmoid(z)
            err = pred - yi
            
            for j in range(4):
                dw[j] += err * xi[j]
            db += err
            
            eps = 1e-15
            loss -= yi * math.log(pred + eps) + (1 - yi) * math.log(1 - pred + eps)
            
        # Update weights
        for j in range(4):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
        loss /= n
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: loss={loss:.6f}")
            
    # Convert to raw weights (divide by 100 since we normalized input)
    w_raw = [wj / 100.0 for wj in w]
    print(f"w_raw = {w_raw}, b_raw = {b}")
    return w_raw, b


def evaluate(records: list[dict], w: list[float], b: float) -> float:
    """Evaluate model accuracy on given records."""
    correct = 0
    for r in records:
        z = sum(w[j-1] * r[f"f{j}"] for j in range(1, 5)) + b
        pred = 1 if sigmoid(z) >= 0.5 else 0
        if pred == r["label"]:
            correct += 1
    return correct / len(records)


def main():
    rng = random.Random(SEED)
    
    print("=" * 60)
    print("  4-Feature Dataset Generation & Model Training")
    print("=" * 60)
    
    print("\nGenerating datasets...")
    train_data = generate_dataset(NUM_TRAIN, rng)
    test_data = generate_dataset(NUM_TEST, rng)
    save_csv(train_data, TRAIN_PATH)
    save_csv(test_data, TEST_PATH)
    
    # Count class distribution
    train_pos = sum(1 for r in train_data if r["label"] == 1)
    test_pos = sum(1 for r in test_data if r["label"] == 1)
    print(f"\nTrain: {train_pos}/{NUM_TRAIN} positive ({100*train_pos/NUM_TRAIN:.1f}%)")
    print(f"Test:  {test_pos}/{NUM_TEST} positive ({100*test_pos/NUM_TEST:.1f}%)")
    
    print("\nTraining logistic regression model...")
    w, b = train_lr(train_data)
    
    print("\nEvaluating model...")
    train_acc = evaluate(train_data, w, b)
    test_acc = evaluate(test_data, w, b)
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    
    print("\nExporting weights...")
    with open(MODEL_PATH, "w") as f:
        f.write(f"Weights: {','.join([str(wj) for wj in w])}\n")
        f.write(f"Bias: {b}\n")
    print(f"Saved weights -> {MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("  Model Summary")
    print("=" * 60)
    print(f"  Features:    4 (f1, f2, f3, f4)")
    print(f"  Classes:     2 (0, 1)")
    print(f"  Weights:     {', '.join([f'{wj:.10f}' for wj in w])}")
    print(f"  Bias:        {b:.10f}")
    print(f"  Test Acc:    {test_acc:.2%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
