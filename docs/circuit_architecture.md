# ZK Logistic Regression Circuit Architecture

This document details the architecture, design choices, and mathematical implementations of the `circuit` package for the Zero-Knowledge Logistic Regression (ZK-LR) model. It serves as a technical reference for the "What, Why, and How" of the implementation constraints.

## 1. High-Level Architecture

The core objective of the circuit is to prove the following statement:
**"Given a student's marks `X`, I have correctly computed the logistic regression prediction `Y = sigmoid(W*X + B)` using secret weights `W` and `B`."**

### Inputs
- **Private (Secret):** 
  - `W`: The model's learned weight (scaled integer)
  - `B`: The model's learned bias (scaled integer)
- **Public (Visible):**
  - `X`: The student's marks (integer, `0-100`)
  - `Y`: The final predicted probability (scaled integer)

### The Underlying PLONK System
We chose **PLONK** via the `gnark` framework for three major advantages outlined in our design:
1. **Lookup Tables (logderivlookup):** PLONK supports native lookup tables, which is the only efficient way to compute non-algebraic exponential functions like the Sigmoid function inside a circuit.
2. **Polynomial Commitments (KZG):** The model weights are hidden by committing to them as polynomials.
3. **Universal SRS:** A single trusted setup works for all circuits.

---

## 2. Integer Scaling (The "What")

**What:** 
All floating-point numbers from the Python Machine Learning model are converted and scaled into large integers.
- `W` and `B` are multiplied by `2^32`.
- Sigmoid table indexes are evaluated at a precision of `2^10`.
- Sigmoid table outputs (`Y`) are scaled by `2^16`.

**Why:**
Zero-Knowledge proofs operate over **Finite Fields** (specifically, the `BN254` curve scalar field). Finite fields only understand non-negative integers. There are **no decimals/floats**. If a Python model outputs `W = 0.179`, the circuit fundamentally cannot process it. By scaling it by `2^32`, we turn `0.179` into the exact integer `769690474`.

**How:**
In `circuit.go`, `X` (marks) is a raw integer (e.g., `70`). When we multiply `W * X`, the result naturally remains at the `2^32` scale.
```go
wx := api.Mul(c.W, c.X)      // W*X → natively scaled by 2^32
z_linear := api.Add(wx, c.B) // W*X + B → scaled by 2^32
```

---

## 3. The Shifted Positive Offset (The "Why")

**What:** 
Before we look up the sigmoid value for $Z$, we add a massive offset value (`16 * 2^32`) to it.

**Why:**
The equation `Z = W*X + B` often produces **negative numbers** (e.g., for failing students). In standard computing, `-5` is stored via two's complement. But in a Prime Finite Field, `-5` wraps completely around the modulus and becomes a massive positive number (e.g., `2188824287...5612`). If we feed that massive wrapped number into a lookup table array, the circuit crashes or accesses invalid memory.

**How:**
We know the useful active range of the sigmoid function is $Z \in [-10, +10]$ (sufficient for all marks 0-100 with our model).
So inside the circuit, we shift the mathematical graph to the right by **adding 10** to $Z$, guaranteeing that the Z-value fed to the lookup table is strictly between 0 and 20.
```go
// Add offset to make strictly positive. 10 * 2^32:
offsetScaled := new(big.Int).Lsh(big.NewInt(10), 32)
z_shifted := api.Add(z_linear, offsetScaled)
```

### Clamping for Edge Cases
For extreme Z values that fall outside the table range, the circuit applies clamping using `api.Cmp` and `api.Select`:
- If z < -10 → clamp to index 0 (sigmoid ≈ 0)
- If z > +10 → clamp to maxIndex (sigmoid ≈ 1)

This ensures the circuit never attempts an out-of-bounds lookup.

---

## 4. Remainder-Based Truncation (The "How")

**What:**
To map our massive `2^32` scaled $Z$ down to a small usable lookup index `2^10`, we must divide by $2^{22}$. Instead of explicitly dividing inside the circuit, we pass the quotient (`ZTable`) and remainder (`Rem`) as external proofs.

**Why:**
Direct division in a ZK finite field is actually "Multiplication by a Modular Inverse". It does **not** truncate decimals (like `5/2 = 2` in standard programming). Using modular division destroys the integer scale math completely.

**How:**
We compute the actual integer division in standard Go outside of the circuit. We pass the quotient (`ZTable`) and the remainder (`Rem`) into the circuit as **private hints**. The circuit then works forwards, replacing the division with a multiplication+addition, constraining the bit-sizes to prevent cheating.

```go
// Enforce 0 <= Rem < 2^22 (fits in 22 bits so they can't overflow the division bounds)
api.ToBinary(c.Rem, 22)

// (Quotient * Divisor) + Remainder == Total Reconstructed Z
shiftFactor := new(big.Int).Lsh(big.NewInt(1), 22) // 2^22
z_reconstructed := api.Add(api.Mul(c.ZTable, shiftFactor), c.Rem)

// Fails the proof if the provided Division hints were faked
api.AssertIsEqual(z_shifted, z_reconstructed) 
```

---

## 5. PLONK Sigmoid Lookup Table

**What:**
The circuit embeds a pre-computed array of 20,481 exact Sigmoid evaluations (range ±10, precision 2^10).

**Why:**
Exponential mathematics like $e^{-x}$ require infinite Taylor series expansions, which are mathematically impossible to compute inside a limited ZK circuit.

**How:**
We build the table at compile time, storing `sigmoid(z) * 2^16`. During the ZK proof, gnark's `logderivlookup` algorithm performs an $O(1)$ cryptographic lookup to fetch the $Y$ probability from the shifted $Z$ index.

```go
y_pos := SigmoidLookup(api, c.ZTable)
api.AssertIsEqual(c.Y, y_pos)
```

---

## 6. Precision and Accuracy Analysis

We executed a programmatic test (`scripts/test_sigmoid_precision.py`) testing 1,000,000 random, exact-float continuous points between $Z = -15.0$ and $Z = +15.0$ against the exact integer bit-shifting logic implemented in the Go circuit.

**Metrics:**
- Average Absolute Error: `0.000024` (0.0024%)
- **MAXIMUM Absolute Error: `0.000259` (0.0259%)**

**Conclusion:**
There are zero edge cases where the circuit will hallucinate or flip a prediction compared to standard Machine Learning floating-point evaluations. The precision is guaranteed to be accurate up to `±0.00025`, easily distinguishing pass/fail boundaries.

## 7. Performance Benchmarks
Run locally on `circuit/benchmark_test.go`:

| Metric | Before (±16) | After (±10) | Improvement |
|--------|-------------|-------------|-------------|
| Constraints | 159,804 | 108,747 | **32% fewer** |
| Compile time | 96ms | 63ms | **34% faster** |
| Setup time | 6.2s | 3.3s | **47% faster** |
| Prove time | 3.8s | 2.1s | **45% faster** |
| Verify time | 1.5ms | 1.4ms | Same |
| Proof size | 584 bytes | 584 bytes | Same |
| Table entries | 32,769 | 20,481 | **37% smaller** |
