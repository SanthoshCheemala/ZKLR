#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def load_model_params(path: Path):
    with path.open("r") as f:
        lines = f.readlines()
    
    coef_line = None
    intercept_line = None
    for line in lines:
        if line.startswith("Coefficient:"):
            coef_line = line
        elif line.startswith("Intercept:"):
            intercept_line = line
    
    if not coef_line or not intercept_line:
        raise ValueError("Could not find Coefficient or Intercept in parameters file")
    
    coef_str = coef_line.split("[[")[1].split("]]")[0].strip()
    coefficient = float(coef_str)
    
    intercept_str = intercept_line.split("[")[1].split("]")[0].strip()
    intercept = float(intercept_str)
    
    return coefficient, intercept


def load_dataset(path: Path):
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "marks": float(row["marks"]),
                "failed": int(row["failed"])
            })
    return rows


def predict(marks, coef, intercept):
    z = coef * marks + intercept
    prob_failed = sigmoid(z)
    predicted_failed = 1 if prob_failed >= 0.5 else 0
    return predicted_failed, prob_failed


def main():
    parser = argparse.ArgumentParser(description="Test dataset with saved logistic regression model")
    parser.add_argument("--data", type=str, default="../data/student_dataset.csv",
                        help="Path to test dataset CSV (default: data/student_dataset.csv)")
    parser.add_argument("--params", type=str, default="../data/best_model_parameters.txt",
                        help="Path to model parameters file (default: data/best_model_parameters.txt)")
    args = parser.parse_args()
    
    param_path = Path(args.params)
    if not param_path.exists():
        print(f"Error: {param_path} not found.")
        sys.exit(1)
    
    coef, intercept = load_model_params(param_path)
    print(f"Loaded model parameters from {param_path}:")
    print(f"  Coefficient: {coef:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print()
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    rows = load_dataset(data_path)
    print(f"Loaded {len(rows)} test samples from {data_path}")
    print()
    
    correct = 0
    predictions = []
    for row in rows:
        marks = row["marks"]
        true_failed = row["failed"]
        pred_failed, prob = predict(marks, coef, intercept)
        predictions.append({
            "marks": marks,
            "true_failed": true_failed,
            "pred_failed": pred_failed,
            "prob_failed": prob
        })
        if pred_failed == true_failed:
            correct += 1
    
    accuracy = 100.0 * correct / len(rows) if rows else 0.0
    
    print(f"Test Results:")
    print(f"  Total samples: {len(rows)}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print()
    
    print("Sample predictions (first 10 rows):")
    print(f"{'Marks':<10} {'True':<10} {'Predicted':<12} {'P(fail)':<10}")
    print("-" * 42)
    for i, pred in enumerate(predictions[:10]):
        true_label = "fail" if pred["true_failed"] == 1 else "pass"
        pred_label = "fail" if pred["pred_failed"] == 1 else "pass"
        print(f"{pred['marks']:<10.2f} {true_label:<10} {pred_label:<12} {pred['prob_failed']:<10.4f}")
    
    tp = sum(1 for p in predictions if p["true_failed"] == 1 and p["pred_failed"] == 1)
    tn = sum(1 for p in predictions if p["true_failed"] == 0 and p["pred_failed"] == 0)
    fp = sum(1 for p in predictions if p["true_failed"] == 0 and p["pred_failed"] == 1)
    fn = sum(1 for p in predictions if p["true_failed"] == 1 and p["pred_failed"] == 0)
    
    print()
    print("Confusion Matrix:")
    print(f"  True Negatives (pass predicted pass): {tn}")
    print(f"  False Positives (pass predicted fail): {fp}")
    print(f"  False Negatives (fail predicted pass): {fn}")
    print(f"  True Positives (fail predicted fail): {tp}")


if __name__ == "__main__":
    main()
