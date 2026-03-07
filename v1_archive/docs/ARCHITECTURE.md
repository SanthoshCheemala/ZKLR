# ZKLR — Detailed System Architecture

> **Zero-Knowledge Logistic Regression**: Privacy-preserving ML inference using PLONK ZK-SNARKs on BN254.

---

## 1. High-Level Overview

ZKLR proves that a logistic regression model (`Y = σ(W·X + B)`) was executed correctly **without revealing the model weights (W, B)** to the verifier/client. The system uses [gnark](https://github.com/ConsenSys/gnark) v0.11 with PLONK over the BN254 elliptic curve.

```mermaid
graph TB
    subgraph ML["ML Pipeline (Python)"]
        GEN["generate_student_dataset.py<br/>3000 samples: marks → pass/fail"]
        TRAIN["train_model.py<br/>Logistic Regression (scikit-learn)"]
        TEST["test_with_saved_model.py<br/>Accuracy validation"]
        GEN --> TRAIN --> TEST
    end

    subgraph ZK["ZK Proof System (Go / gnark)"]
        SINGLE["main.go<br/>Single-Sample Parallel Prover"]
        BATCH["main_batched.go<br/>Batched Prover (20 samples/proof)"]
        RECUR["cmd/recursive_accuracy/<br/>Batch Accuracy + Aggregator"]
        SEC["cmd/security_demo/<br/>Security Property Tests"]
        SIM["sim/ + simulation/<br/>Network Simulation"]
    end

    subgraph DATA["Data & Artifacts"]
        CSV["student_dataset.csv"]
        MODEL["Model params: W, B"]
        CACHE["Circuit caches (.bin)"]
        METRICS["metrics_output/ (PNG, CSV, JSON)"]
    end

    ML --> DATA
    DATA --> ZK
    ZK --> METRICS
```

---

## 2. Core Cryptographic Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Proof System** | PLONK | Universal zkSNARK with polynomial commitments |
| **Elliptic Curve** | BN254 | 128-bit security, ~254-bit scalar field |
| **Commitment** | KZG (Kate) | Polynomial commitment scheme |
| **Constraint System** | SparseR1CS | gnark's optimized PLONK constraint format |
| **Lookup Tables** | Log-derivative | Efficient table lookups for sigmoid |
| **Library** | gnark v0.11 + gnark-crypto v0.14 | Consensys Go ZK framework |

### Trusted Setup

Currently uses `unsafekzg.NewSRS()` (deterministic, for development only). Production requires an MPC ceremony.

---

## 3. Fixed-Point Arithmetic System

All computations inside circuits use fixed-point integers:

```mermaid
graph LR
    subgraph Formats["Fixed-Point Formats"]
        Q32["Q32 (2³²)<br/>W, B, X — linear computation"]
        Q10["Q10 (2¹⁰)<br/>Z — sigmoid LUT index"]
        Q16["Q16 (2¹⁶)<br/>Y — sigmoid output"]
    end

    Q32 -- "Z = (W·X + B) >> 22" --> Q10
    Q10 -- "LUT[Z] → sigmoid(Z)" --> Q16
```

| Format | Scale Factor | Precision | Used For |
|--------|-------------|-----------|----------|
| **Q32** | 2³² = 4,294,967,296 | ~9.3 decimal digits | Weight (W), Bias (B), Input (X) |
| **Q10** | 2¹⁰ = 1,024 | ~3 decimal digits | Sigmoid lookup index |
| **Q16** | 2¹⁶ = 65,536 | ~4.8 decimal digits | Sigmoid output / prediction |

**Conversion**: `Z_Q10 = Z_Q32 / 2^(32-10) = Z_Q32 >> 22`

---

## 4. Circuit Architecture

### 4.1 Single-Sample Circuit (`main.go` → `Circuit`)

The foundational circuit proving `Y = sigmoid(W·X + B)` for one sample.

```mermaid
graph TD
    subgraph Private["Private Witness"]
        W["W (weight, Q32)"]
        B["B (bias, Q32)"]
        Z["Z (linear output, Q10)"]
    end

    subgraph Public["Public Inputs"]
        X["X (marks, Q32)"]
        Y["Y (prediction, Q16)"]
    end

    subgraph Circuit["Circuit Logic"]
        LUT["Sigmoid LUT<br/>8193 entries<br/>range [0, 8]"]
        CLAMP["Clamping Logic<br/>|z| > 8 → saturate"]
        NEG["Symmetry<br/>σ(-z) = 1 - σ(z)"]
        ASSERT["AssertIsEqual<br/>result == Y"]
    end

    Z --> CLAMP
    CLAMP --> LUT
    LUT --> NEG
    NEG --> ASSERT
    Y --> ASSERT

    style Private fill:#2d1f3d,stroke:#7c3aed,color:#fff
    style Public fill:#1a3a2a,stroke:#22c55e,color:#fff
```

**Constraint count**: ~170,000 (dominated by the log-derivative lookup table)

**Key design decision**: `Z = W·X + B` is pre-computed outside the circuit and passed as a private witness. This avoids division-by-negative issues in the finite field.

### 4.2 Sigmoid Lookup Table

```
Index i ∈ [0, 8192]
  → x = i / 2¹⁰           (float in [0, 8])
  → y = 1 / (1 + e⁻ˣ)     (sigmoid)
  → scaled = y × 2¹⁶       (Q16 integer)
  → table.Insert(scaled)
```

**Negative handling**: For `z < 0`, use the identity `σ(-z) = 1 - σ(|z|)`.

**Saturation**: If `|z| > 8`, output is clamped to 0 (negative) or 65,535 (positive).

### 4.3 Batch Circuit (`main_batched.go` → `BatchCircuit`)

Processes **20 samples** in a single proof by sharing one LUT across all samples.

```mermaid
graph TB
    subgraph Shared["Shared (Private)"]
        W2["W (weight)"]
        B2["B (bias)"]
        LUT2["Single Sigmoid LUT<br/>(shared across all 20)"]
    end

    subgraph PerSample["Per-Sample Arrays (×20)"]
        X2["X[20] (public)"]
        Z2["Z[20] (private)"]
        Y2["Y[20] (public)"]
    end

    W2 --> LUT2
    X2 --> LUT2
    Z2 --> LUT2
    LUT2 --> Y2
```

**Key benefit**: The LUT is compiled once into the circuit, so batching amortizes the LUT overhead (~58K constraints) across 20 samples.

### 4.4 Batch Accuracy Circuit (`cmd/recursive_accuracy/` → `BatchAccuracyCircuit`)

Extends the batch circuit with **accuracy counting** — proves the sigmoid prediction AND compares it against actual labels.

| Field | Visibility | Type | Purpose |
|-------|-----------|------|---------|
| `W` | Private | Variable | Model weight |
| `B` | Private | Variable | Model bias |
| `X[50]` | Public | Array | Input features |
| `Z[50]` | Private | Array | Linear outputs |
| `Y[50]` | Public | Array | Sigmoid predictions |
| `ActualLabels[50]` | Public | Array | Ground truth labels |
| `CorrectCount` | Public | Variable | Number of correct predictions |

**Logic**: For each sample, it computes sigmoid, thresholds at 0.5 (Q16 = 32,768), compares predicted vs actual label, and increments `CorrectCount`. Finally asserts the claimed `CorrectCount` matches the circuit's computed value.

### 4.5 Aggregator Circuit (`AggregatorCircuit`)

The **top-level recursive proof** that aggregates batch results and enforces the accuracy threshold.

```mermaid
graph TB
    B1["Batch 1: correct=48"] --> AGG
    B2["Batch 2: correct=49"] --> AGG
    B3["..."] --> AGG
    B60["Batch 60: correct=50"] --> AGG

    subgraph AGG["Aggregator Circuit"]
        SUM["Sum all BatchCorrectCounts[60]"]
        CHECK1["Assert TotalSamples == 3000"]
        CHECK2["Assert TotalCorrect ≥ 2910<br/>(97% threshold)"]
    end

    AGG --> PROOF["Single Aggregator Proof<br/>~900 bytes"]
```

**Constraint count**: ~5,388 (lightweight — just integer sums and comparisons)

### 4.6 Security Demo Circuits (`cmd/security_demo/`)

Two circuits used for security property demonstrations:

1. **`Circuit`** (same as main) — used for correctness, tampered proof, tampered input tests
2. **`SimpleCircuit`** (`Y = X²`) — used for zero-knowledge, wrong witness, wrong VK tests

**Security tests performed**:
- ✅ **Completeness**: Honest prover succeeds
- ✅ **Soundness**: Tampered proofs/inputs rejected
- ✅ **Zero-Knowledge**: Proof reveals nothing about witness
- ✅ **Wrong Witness**: Invalid witness rejected
- ✅ **Wrong VK**: Cross-circuit verification fails

---

## 5. Execution Modes

```mermaid
graph TD
    subgraph Mode1["Mode 1: Single-Sample Parallel"]
        M1["main.go"]
        M1A["3000 individual proofs<br/>Worker pool (goroutines)<br/>Each: prove + verify"]
    end

    subgraph Mode2["Mode 2: Batched"]
        M2["main_batched.go / main_batched_flexible.go"]
        M2A["150 batch proofs (20 samples each)<br/>Sequential batch processing<br/>Each: prove + verify"]
    end

    subgraph Mode3["Mode 3: Recursive Accuracy"]
        M3["cmd/recursive_accuracy/"]
        M3A["60 batch proofs (50 samples each)<br/>+ 1 aggregator proof<br/>Proves accuracy ≥ 97%"]
    end

    subgraph Mode4["Mode 4: Security Demo"]
        M4["cmd/security_demo/"]
        M4A["6 security property tests<br/>+ Performance evaluation<br/>+ Metrics export (JSON/CSV)"]
    end

    subgraph Mode5["Mode 5: Network Simulation"]
        M5["sim/ → simulation/"]
        M5A["4-phase distributed sim<br/>Client ↔ Server flow<br/>Animated or actual proofs"]
    end
```

### Mode 1: Single-Sample Parallel (`main.go`)

```
Setup → Compile Circuit → Generate SRS → PLONK Setup (PK, VK)
  ↓
Load 3000 samples → Create ProofTasks
  ↓
Spawn worker goroutines (1 currently, designed for N CPUs)
  ↓
Each worker: compute witness → plonk.Prove → plonk.Verify
  ↓
Collect results → Export metrics
```

**Caching**: Circuit (CCS), proving key (PK), and verifying key (VK) are serialized to `data/cache/` for reuse.

### Mode 2: Batched (`main_batched.go`)

Same pipeline but groups 20 samples into a single `BatchCircuit` proof. Padding is used when total samples aren't divisible by batch size.

### Mode 3: Recursive Accuracy (`cmd/recursive_accuracy/`)

```
Phase 1: Compile BatchAccuracyCircuit + AggregatorCircuit
Phase 2: Generate 60 batch proofs (50 samples each), collecting CorrectCount
Phase 3: Feed all 60 CorrectCounts into AggregatorCircuit, prove ≥ 97%
Phase 4: Verify the single aggregator proof
```

---

## 6. Data Flow

```mermaid
sequenceDiagram
    participant PY as Python Scripts
    participant CSV as student_dataset.csv
    participant GO as Go Prover
    participant CACHE as data/cache/
    participant OUT as metrics_output/

    PY->>CSV: Generate 3000 samples (marks, failed)
    PY->>CSV: Train model → W=-0.857, B=50.947
    GO->>CSV: Load samples
    GO->>CACHE: Load/save compiled circuits
    GO->>GO: For each sample/batch:<br/>1. Compute Z = W·X + B (Q32→Q10)<br/>2. Compute Y = sigmoid(Z) (Q16)<br/>3. Build witness<br/>4. plonk.Prove()<br/>5. plonk.Verify()
    GO->>OUT: Export metrics (TXT, CSV, JSON, PNG)
```

### ML Model Parameters

Hardcoded in Go after Python training:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **W** (weight) | -0.85735312 | Logistic regression coefficient |
| **B** (bias) | 50.94705066 | Logistic regression intercept |
| **Decision boundary** | ~59.4 marks | Students below this score → fail |

---

## 7. Project Directory Map

```
BTP_Project/
├── main.go                         # Mode 1: Single-sample parallel prover (645 lines)
├── main_batched.go                 # Mode 2: Batched prover, BatchSize=20 (450 lines)
├── main_batched_flexible.go        # Mode 2 variant (identical to main_batched.go)
│
├── cmd/
│   ├── recursive_accuracy/
│   │   └── main.go                 # Mode 3: BatchAccuracy(50) + Aggregator(60) (459 lines)
│   └── security_demo/
│       ├── main.go                 # Mode 4: 6 security tests + perf eval (1365 lines)
│       └── export.go               # Metrics export: JSON, CSV, plot scripts (913 lines)
│
├── sim/
│   └── main.go                     # Mode 5 entry: --animated or actual proofs (34 lines)
├── simulation/
│   └── network.go                  # Mode 5: 4-phase distributed simulation (164 lines)
│
├── lib/
│   └── version.go                  # Project metadata: Name="ZKLR", Version="0.1.0"
│
├── scripts/
│   ├── generate_student_dataset.py # Generates labeled student data (162 lines)
│   ├── train_model.py              # Trains logistic regression model (82 lines)
│   └── test_with_saved_model.py    # Validates model against dataset (134 lines)
│
├── data/
│   ├── student_dataset.csv         # 3000 rows: marks, failed (0 or 1)
│   ├── student_dataset_test.csv    # Test split
│   ├── cache/                      # Compiled circuits, keys, metrics
│   └── README.md                   # Data layout documentation
│
├── metrics_output/                 # Generated visualizations and analysis
│   ├── *.png                       # Benchmark charts (7+ plots)
│   ├── *.csv                       # Raw metric data
│   ├── *.json                      # Full analysis data
│   ├── metrics_collect*.py         # Metric collection scripts
│   └── plot_*.py                   # Visualization scripts
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md    # Project documentation
│
├── go.mod                          # Module: github.com/santhoshcheemala/ZKLR
├── go.sum
├── updated_results_chapter.tex     # LaTeX results chapter
└── README.md                       # Project overview and usage guide
```

---

## 8. Proof Lifecycle (Detailed)

```mermaid
stateDiagram-v2
    [*] --> Compile: frontend.Compile()
    Compile --> SRS: unsafekzg.NewSRS()
    SRS --> Setup: plonk.Setup()
    Setup --> Ready

    state Ready {
        [*] --> BuildWitness
        BuildWitness --> ComputeZ: Z = W·X + B (Q32→Q10)
        ComputeZ --> ComputeY: Y = sigmoid(Z) (Q16)
        ComputeY --> NewWitness: frontend.NewWitness()
        NewWitness --> Prove: plonk.Prove(ccs, pk, witness)
        Prove --> Verify: plonk.Verify(proof, vk, publicWitness)
        Verify --> [*]
    }
```

### Step-by-step for a single sample (marks=75):

1. **Scale input**: `X = 75 × 2³² = 322,122,547,200`
2. **Linear computation**: `Z_Q32 = W_Q32 × X_Q32 / 2³² + B_Q32`
3. **Rescale**: `Z_Q10 = Z_Q32 / 2²² ≈ -13,352` (negative → student likely passes)
4. **Sigmoid LUT lookup**: Index `|Z_Q10|`, apply symmetry for negative Z
5. **Output**: `Y_Q16 ≈ 65535` (probability ≈ 1.0 → pass)
6. **Circuit**: Verifies the above computation without revealing W or B

---

## 9. Security Model

```mermaid
graph LR
    subgraph Server["Server (Prover)"]
        MW["Knows W, B"]
        PROOF["Generates proofs"]
    end

    subgraph Client["Client (Verifier)"]
        VK["Has Verifying Key"]
        VERIFY["Verifies proofs"]
        DATASET["Has test dataset"]
    end

    Server -- "Proof + Public Y" --> Client
    Client -- "Sample X" --> Server

    style Server fill:#2d1f3d,stroke:#7c3aed,color:#fff
    style Client fill:#1a3a2a,stroke:#22c55e,color:#fff
```

| Property | Guarantee |
|----------|-----------|
| **Completeness** | Honest prover with correct computation always produces valid proof |
| **Soundness** | Prover cannot create valid proof for incorrect computation |
| **Zero-Knowledge** | Proof reveals nothing about W, B, or Z |
| **Accuracy Threshold** | Aggregator proof guarantees ≥97% accuracy on full dataset |

---

## 10. Dependencies

```mermaid
graph BT
    MAIN["main.go / main_batched.go"]
    RECUR["cmd/recursive_accuracy"]
    SEC["cmd/security_demo"]
    SIM["sim/ + simulation/"]

    GNARK["gnark v0.11<br/>(frontend, backend/plonk,<br/>constraint/bn254, logderivlookup)"]
    CRYPTO["gnark-crypto v0.14<br/>(ecc, BN254)"]
    SKLEARN["scikit-learn<br/>(LogisticRegression)"]

    MAIN --> GNARK
    MAIN --> CRYPTO
    RECUR --> GNARK
    RECUR --> CRYPTO
    SEC --> GNARK
    SEC --> CRYPTO
    SIM -.-> GNARK

    subgraph Python
        GEN2["generate_student_dataset.py"] --> SKLEARN
        TRAIN2["train_model.py"] --> SKLEARN
    end
```

### Go Dependencies (from `go.mod`)

| Package | Version | Purpose |
|---------|---------|---------|
| `gnark` | v0.11.0 | ZK-SNARK framework (circuits, PLONK prover/verifier) |
| `gnark-crypto` | v0.14.0 | Cryptographic primitives (BN254, KZG, hashing) |

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Logistic regression training |
| `numpy` | Numerical computation |

---

## 11. Performance Characteristics

| Metric | Single-Sample | Batched (20) | Recursive (50+Agg) |
|--------|--------------|-------------|-------------------|
| **Constraints** | ~170K | ~170K × 20 amortized | ~1.6M per batch + 5.4K agg |
| **Proofs for 3000 samples** | 3000 | 150 | 60 + 1 |
| **Proof size** | ~900 bytes each | ~900 bytes each | ~900 bytes (aggregator) |
| **Verification** | ~1.3ms each | ~1.3ms each | ~1.3ms (aggregator only) |
| **Key advantage** | Simple, parallelizable | Amortized LUT cost | Single proof for accuracy |

---

*Generated from codebase analysis on 2026-02-28*
