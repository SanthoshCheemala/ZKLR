#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np
except ImportError:
    print("Error: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)


def load_dataset(path: Path):
    marks_list = []
    failed_list = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            marks_list.append(int(float(row["marks"])))
            failed_list.append(int(row["failed"]))
    return np.array(marks_list).reshape(-1, 1), np.array(failed_list)


def main():
    data_path = Path("../data/student_dataset.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    X, y = load_dataset(data_path)
    print(f"Loaded {len(X)} training samples from {data_path}")
    print(f"  Pass (failed=0): {np.sum(y == 0)}")
    print(f"  Fail (failed=1): {np.sum(y == 1)}")
    print()
    
    print("Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    print(f"Model parameters:")
    print(f"  Coefficient: {coef:.8f}")
    print(f"  Intercept: {intercept:.8f}")
    print()
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Training accuracy: {accuracy * 100:.2f}%")
    print()
    
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives (pass predicted pass): {cm[0][0]}")
    print(f"  False Positives (pass predicted fail): {cm[0][1]}")
    print(f"  False Negatives (fail predicted pass): {cm[1][0]}")
    print(f"  True Positives (fail predicted fail): {cm[1][1]}")
    print()
    
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=["pass", "fail"]))
    
    param_path = Path("data/best_model_parameters.txt")
    with param_path.open("w") as f:
        f.write(f"Coefficient: [[{coef:.8f}]]\n")
        f.write(f"Intercept: [{intercept:.8f}]\n")
    
    print(f"Model parameters saved to {param_path}")
    
    decision_boundary = -intercept / coef
    print(f"\nDecision boundary (marks): {decision_boundary:.2f}")
    print(f"  Students with marks < {decision_boundary:.2f} are predicted to fail")
    print(f"  Students with marks >= {decision_boundary:.2f} are predicted to pass")


if __name__ == "__main__":
    main()
