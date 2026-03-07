"""
generate_bmi_dataset.py — Generate height+weight dataset and train LR model.

Labels:
  - BMI ≥ 25 → overweight (1)
  - BMI <  25 → normal    (0)

BMI = weight(kg) / height(m)^2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

# ─── Config ──────────────────────────────────────────────────
N_TRAIN = 3000
N_TEST  = 500

# ─── Generate Dataset ────────────────────────────────────────
def generate_samples(n):
    heights = np.random.normal(170, 10, n)   # cm, mean=170, std=10
    heights = np.clip(heights, 140, 210)

    weights = np.random.normal(70, 15, n)    # kg, mean=70, std=15
    weights = np.clip(weights, 40, 150)

    # BMI = weight(kg) / (height(m))^2
    bmi = weights / (heights / 100.0) ** 2

    # Label: 1=overweight (BMI≥25), 0=normal
    labels = (bmi >= 25).astype(int)

    return heights, weights, bmi, labels

print("Generating datasets...")
h_train, w_train, bmi_train, y_train = generate_samples(N_TRAIN)
h_test,  w_test,  bmi_test,  y_test  = generate_samples(N_TEST)

# Save raw CSVs (unscaled integers for ZK circuit: round height to int, weight×10 for 1 decimal)
os.makedirs("data", exist_ok=True)

train_df = pd.DataFrame({
    "height_cm": h_train.astype(int),    # integer cm e.g. 172
    "weight_kg": (w_train * 10).astype(int),  # ×10 for 1 decimal e.g. 735 = 73.5kg
    "overweight": y_train
})

test_df = pd.DataFrame({
    "height_cm": h_test.astype(int),
    "weight_kg": (w_test * 10).astype(int),
    "overweight": y_test
})

train_df.to_csv("data/bmi_dataset_train.csv", index=False)
test_df.to_csv("data/bmi_dataset_test.csv",  index=False)

print(f"Train: {N_TRAIN} samples  |  Test: {N_TEST} samples")
print(f"Train overweight rate: {y_train.mean()*100:.1f}%")
print(f"Test  overweight rate: {y_test.mean()*100:.1f}%")

# ─── Train LR Model (on raw integer features) ────────────────
# Use the same integer features the ZK circuit will use
X_train = np.column_stack([h_train.astype(int), (w_train * 10).astype(int)])
X_test  = np.column_stack([h_test.astype(int),  (w_test  * 10).astype(int)])

# Logistic Regression without scaler (ZK uses raw integers)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== Model Results ===")
print(f"Accuracy: {acc*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Overweight"]))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── Extract Weights ─────────────────────────────────────────
w1 = model.coef_[0][0]   # weight for height_cm
w2 = model.coef_[0][1]   # weight for weight_kg (×10)
b  = model.intercept_[0]

print(f"\n=== Model Weights ===")
print(f"w1 (height): {w1:.10f}")
print(f"w2 (weight): {w2:.10f}")
print(f"b  (bias):   {b:.10f}")

# Scale for ZK circuit (×2^32)
SCALE = 2**32
w1_scaled = int(w1 * SCALE)
w2_scaled = int(w2 * SCALE)
b_scaled  = int(b  * SCALE)

print(f"\n=== ZK Circuit Constants (×2^32) ===")
print(f"W1_SCALED = {w1_scaled}")
print(f"W2_SCALED = {w2_scaled}")
print(f"B_SCALED  = {b_scaled}")

# ─── Save Model Weights ───────────────────────────────────────
with open("data/bmi_model_weights.txt", "w") as f:
    f.write("# ZK-LR BMI Model Weights (2-feature)\n")
    f.write("# Features: height_cm (int), weight_kg×10 (int)\n")
    f.write("# Label: 1=overweight (BMI≥25), 0=normal\n")
    f.write(f"# Trained on {N_TRAIN} samples | Test accuracy: {acc*100:.2f}%\n\n")
    f.write(f"# Float values\n")
    f.write(f"w1 = {w1:.10f}   # height coefficient\n")
    f.write(f"w2 = {w2:.10f}   # weight coefficient\n")
    f.write(f"b  = {b:.10f}    # bias\n\n")
    f.write(f"# Pre-scaled for Go ZK circuit (× 2^32)\n")
    f.write(f"W1_SCALED = {w1_scaled}\n")
    f.write(f"W2_SCALED = {w2_scaled}\n")
    f.write(f"B_SCALED  = {b_scaled}\n")

print(f"\nSaved:")
print(f"  data/bmi_dataset_train.csv  ({N_TRAIN} rows)")
print(f"  data/bmi_dataset_test.csv   ({N_TEST} rows)")
print(f"  data/bmi_model_weights.txt")
print(f"\n✅ Done! Next: update ZK circuit for 2-feature logistic regression.")

# ─── Quick sanity check ───────────────────────────────────────
print(f"\n=== Sample Predictions (integer inputs) ===")
print(f"{'Height':>8} {'Weight×10':>10} {'BMI':>6} {'Actual':>8} {'Pred':>8}")
for i in range(8):
    h = h_test[i].astype(int)
    wt = (w_test[i]*10).astype(int)
    bmi_val = bmi_test[i]
    actual = "OW" if y_test[i] else "Normal"
    pred_v = "OW" if y_pred[i] else "Normal"
    print(f"{h:>8} {wt:>10} {bmi_val:>6.1f} {actual:>8} {pred_v:>8}")
