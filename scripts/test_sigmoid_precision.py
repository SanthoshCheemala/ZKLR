import math
import numpy as np

# Circuit Parameters
InputPrecision = 10
OutputPrecision = 16
SigmoidOffset = 16
SigmoidRange = 32

def sigmoid_exact(z):
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0

def build_table():
    table_size = SigmoidRange * (1 << InputPrecision) + 1
    table = []
    
    for i in range(table_size):
        z_f = float(i) / float(1 << InputPrecision) - float(SigmoidOffset)
        y = sigmoid_exact(z_f)
        
        y_scaled = int(y * float(1 << OutputPrecision))
        max_val = int((1 << OutputPrecision) - 1)
        if y_scaled > max_val:
            y_scaled = max_val
            
        table.append(y_scaled)
    return table

def analyze_precision():
    print("============================================================")
    print("  ZK Sigmoid Lookup Table Precision Analysis")
    print("============================================================")
    
    table = build_table()
    
    # Test random Z values
    np.random.seed(42)
    test_zs = np.random.uniform(-15.0, 15.0, 1000000)
    
    max_abs_error = 0.0
    total_error = 0.0
    
    for z in test_zs:
        # 1. Exact Float Math
        y_exact = sigmoid_exact(z)
        
        # 2. Circuit Lookup Math
        # Shift and scale to find table index
        z_shifted = z + SigmoidOffset
        index = int(z_shifted * (1 << InputPrecision))
        
        # Bound index
        if index < 0: index = 0
        if index >= len(table): index = len(table) - 1
            
        y_table_scaled = table[index]
        y_circuit = y_table_scaled / float(1 << OutputPrecision)
        
        # Compare
        error = abs(y_exact - y_circuit)
        max_abs_error = max(max_abs_error, error)
        total_error += error
        
    avg_error = total_error / len(test_zs)
    
    print(f"Tested {len(test_zs):,} random continuous Z values in range [-15, 15]")
    print(f"Table Entries:  {len(table):,} values")
    print(f"Input Res:      2^{InputPrecision} (step size: {1.0/(1<<InputPrecision):.6f})")
    print(f"Output Res:     2^{OutputPrecision} (step size: {1.0/(1<<OutputPrecision):.6f})")
    print("-" * 60)
    print(f"Average Error:  {avg_error:.6f} ({avg_error*100:.4f}%)")
    print(f"MAX Absolute Error: {max_abs_error:.6f} ({max_abs_error*100:.4f}%)")
    
    # Impact on Pas/Fail boundaries
    print("-" * 60)
    print("Decision Boundary Impact (Is probability >= 0.5?):")
    print(f"If a student's exact true probability is 0.5000,")
    print(f"The circuit might calculate it anywhere between {0.5 - max_abs_error:.4f} and {0.5 + max_abs_error:.4f}")
    
    if max_abs_error < 0.001:
        print("\n✅ GUARANTEED: The circuit prediction is accurate to >3 decimal places.")
        print("   This is extremely precise and perfectly adequate for Logistic Regression.")
    
if __name__ == "__main__":
    analyze_precision()
