#!/usr/bin/env python3
"""
Generate synthetic dataset with 4 features and 2 classes (0 or 1).
Train a logistic regression model and export weights.

Usage:
    python3 prepare_4f_model.py [--train N] [--test N1,N2,N3]
"""
import csv
import math
import random
import argparse
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_dataset(n, rng):
    """Generate dataset with 4 features and binary class label."""
    records = []
    for _ in range(n):
        f = [rng.randint(0, 100) for _ in range(4)]
        total = sum(f)
        z = (total - 200) / 40.0
        p = 1.0 / (1.0 + math.exp(-z))
        label = 1 if rng.random() < p else 0
        records.append({"f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3], "label": label})
    return records


def save_csv(records, path):
    """Save records to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["f1", "f2", "f3", "f4", "label"])
        writer.writeheader()
        writer.writerows(records)
    print("Saved {} records -> {}".format(len(records), path))


def sigmoid(z):
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train_lr(records, lr=0.5, epochs=2000):
    """Train logistic regression with gradient descent."""
    X = [[r["f{}".format(i)]/100.0 for i in range(1, 5)] for r in records]
    Y = [r["label"] for r in records]
    n = len(X)
    
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
            
        for j in range(4):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
        loss /= n
        
        if (epoch + 1) % 500 == 0:
            print("Epoch {}: loss={:.6f}".format(epoch+1, loss))
            
    w_raw = [wj / 100.0 for wj in w]
    return w_raw, b


def evaluate(records, w, b):
    correct = 0
    for r in records:
        z = sum(w[j-1] * r["f{}".format(j)] for j in range(1, 5)) + b
        pred = 1 if sigmoid(z) >= 0.5 else 0
        if pred == r["label"]:
            correct += 1
    return correct / len(records)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset and train model")
    parser.add_argument("--train", type=int, default=10000, help="Training samples (default: 10000)")
    parser.add_argument("--test", type=str, default="1000,2000,3000", help="Test sizes comma-separated (default: 1000,2000,3000)")
    args = parser.parse_args()
    
    test_sizes = [int(x.strip()) for x in args.test.split(",")]
    
    rng = random.Random(SEED)
    
    print("=" * 60)
    print("  4-Feature Dataset Generation & Model Training")
    print("=" * 60)
    
    # Generate training data
    print("\nGenerating {} training samples...".format(args.train))
    train_data = generate_dataset(args.train, rng)
    train_path = DATA_DIR / "synthetic_4f_train.csv"
    save_csv(train_data, train_path)
    
    train_pos = sum(1 for r in train_data if r["label"] == 1)
    print("Train: {}/{} positive ({:.1f}%)".format(train_pos, args.train, 100*train_pos/args.train))
    
    # Train model
    print("\nTraining logistic regression model...")
    w, b = train_lr(train_data)
    
    train_acc = evaluate(train_data, w, b)
    print("Train Accuracy: {:.2%}".format(train_acc))
    
    # Save weights
    model_path = DATA_DIR / "model_weights.txt"
    with open(model_path, "w") as f:
        f.write("Weights: {}\n".format(','.join([str(wj) for wj in w])))
        f.write("Bias: {}\n".format(b))
    print("Saved weights -> {}".format(model_path))
    
    # Generate test datasets of different sizes
    print("\nGenerating test datasets...")
    for size in test_sizes:
        test_data = generate_dataset(size, rng)
        test_path = DATA_DIR / "test_{}.csv".format(size)
        save_csv(test_data, test_path)
        
        test_acc = evaluate(test_data, w, b)
        print("  Test {}: {:.2%} accuracy".format(size, test_acc))
    
    print("\n" + "=" * 60)
    print("  Model Summary")
    print("=" * 60)
    print("  Features:    4 (f1, f2, f3, f4)")
    print("  Classes:     2 (0, 1)")
    print("  Train size:  {}".format(args.train))
    print("  Test sizes:  {}".format(test_sizes))
    print("  Weights:     {}".format(', '.join(['{:.10f}'.format(wj) for wj in w])))
    print("  Bias:        {:.10f}".format(b))
    print("=" * 60)


if __name__ == '__main__':
    main()
