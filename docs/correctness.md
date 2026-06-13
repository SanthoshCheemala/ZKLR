# ZKLR Correctness & Soundness

This document states precisely what a ZKLR proof claims, argues the soundness
of each circuit gadget, derives the fixed-point fidelity bound, and maps every
claim to the test that checks it empirically.

Constants (`circuit/circuit.go`, `circuit/sigmoid.go`):
`P = 2^32` (weight scale), `Q = 2^10` (table index scale), `R = 2^16` (output
scale), `Δ = 2^22 = P/Q` (truncation factor), `M = 1000` (ModelOffset),
`S = 10` (SigmoidOffset). All arithmetic is over the BN254 scalar field
(`p ≈ 2^254`).

## 1. The proven relation

### 1.1 Label mode (default), batch size B, F features

Public instance: `x ∈ ([0, 2^12) ∩ ℤ)^{B×F}`, `ℓ ∈ {0,1}^B`, commitment `C`.
Private witness: `w ∈ ℤ^F`, `b ∈ ℤ` (scaled by `P`), auxiliary `(t_i, r_i)` per sample.

A proof is valid iff there exist `w, b, t, r` such that for every sample `i`:

```
(R1)  C = MiMC(w_1, ..., w_F, b)                                  — model binding
(R2)  x_{i,j} ∈ [0, 2^12)            for all j                    — input range
(R3)  z_i := Σ_j w_j·x_{i,j} + b + M·P                            — linear layer
(R4)  z_i = t_i·Δ + r_i,  r_i ∈ [0, 2^22),  t_i ∈ [0, 2^24)       — truncation
(R5)  u_i := clamp(t_i, (M−S)·Q, (M+S)·Q) − (M−S)·Q               — table window
(R6)  σ_i := T[u_i]      where T[k] = ⌊sigmoid(k/Q − S)·R⌋        — lookup
(R7)  ℓ_i = bit_{15}(σ_i)                                         — label bit
```

In prob mode, (R7) is replaced by the public output `y_i = σ_i`.
In one-vs-rest mode (C classes), (R1) commits to all C weight rows and biases
class-major, and (R3)–(R7) hold per (sample, class) pair against one shared T.

**What is NOT claimed:** that the model is accurate, fair, or was trained on
any particular data — only that the *committed* model was *correctly evaluated*.

## 2. Gadget soundness arguments

### 2.1 Truncation (R4) — unique decomposition

Claim: given the range checks, `(t_i, r_i)` is the unique decomposition of
`z_i`, so the prover cannot steer the lookup index.

The constraints force `r_i < 2^22` and `t_i < 2^24` (via `ToBinary`), hence
`t_i·Δ + r_i < 2^46 + 2^22 ≪ p`: no field wrap-around can occur, so (R4) is
an equation over ℤ. For integers, `z = t·Δ + r` with `0 ≤ r < Δ` is Euclidean
division — `t` and `r` are unique. Conversely (R4) bounds the honest domain:
`z_i` must lie in `[0, 2^46)`, i.e. the model output `Σw·x + b` must lie in
`(−M·P, 2^46 − M·P)`; the completeness domain `|Σw·x/P + b/P| < M = 1000`
satisfies this with huge margin. The 24-bit bound on `t_i` also makes
`api.Cmp` in (R5) sound (its inputs are far below the comparator's limit).

Empirical: `TestWrongRemRejected`, `TestInvalidZ`.

### 2.2 Clamp (R5) — saturation cannot flip a label

For `t_i` outside the window `[(M−S)Q, (M+S)Q]` the circuit substitutes the
window edge, i.e. evaluates `sigmoid(±10)`. Since
`sigmoid(10) = 1 − 4.54·10⁻⁵` and the true value at any `z > 10` is closer to
1, clamping changes the probability by less than `4.54·10⁻⁵` and never moves
it across the 0.5 threshold (which would require `|z| < ~10⁻⁴`, deep inside
the window). So saturation affects neither soundness nor the label.

Empirical: `TestClampSaturation` (both branches).

### 2.3 Commitment (R1) — model binding

MiMC over BN254 is collision-resistant; finding `(w', b') ≠ (w, b)` with the
same `C` is a collision. Hence all proofs under a published `C` were computed
with the same scaled weights, across batches and across time. The off-circuit
computation (`circuit.ComputeCommitment`) reduces negative integers into the
field exactly as gnark does for witnesses, so honest provers always match.

Empirical: `TestWrongCommitmentRejected`, `TestModelSwapRejected`,
`TestBatchWrongCommitmentRejected`, `TestOneVsRestClassSwapRejected`,
`TestCommitmentDeterministic`.

### 2.4 Label bit (R7) — extraction resistance

`σ_i < R = 2^16` (table values are clamped to `R−1`), so the 16-bit
decomposition exists and `bit_15(σ_i) = 1 ⇔ σ_i ≥ 2^15 ⇔ p_i ≥ 0.5`. Only
this bit is public. With exact probabilities public (prob mode), an observer
solves `logit(y) = w·x + b` from F+1 samples and recovers the model (Tramèr
et al., USENIX Security 2016); with one bit per sample, recovery requires
active decision-boundary search, which the model owner can rate-limit. This
is a leakage reduction, not elimination — stated as such in the threat model.

Empirical: `TestBatchLabelFlippedRejected`, `TestOneVsRestFlippedLabelRejected`.

### 2.5 Protocol properties

Completeness, knowledge soundness and zero-knowledge are inherited from
PLONK [GWC19] with the log-derivative lookup argument [Hab22] as implemented
by gnark v0.11 (BN254, KZG commitments). Trusted-setup assumptions:
`unsafekzg` keys are for development only; production keys must derive from a
public ceremony transcript (see `docs/TRUSTED_SETUP.md`, `prover.LoadCeremonySRS`,
validated by `TestLoadCeremonySRS`).

## 3. Fixed-point fidelity bound

Sources of deviation between the circuit output and the float model
`p = sigmoid(w·x + b)`, using `max|sigmoid'| = 1/4`:

| source | magnitude | effect on p |
|---|---|---|
| weight/bias truncation to `1/P` grid | `|Δz| ≤ (Σ_j x_j + 1)/P ≤ (F·2^12+1)/2^32` | `≤ 2.2·10⁻⁵/4` (worst dataset: credit, F=23) |
| index truncation to `1/Q` grid (R4) | `Δz ∈ [0, 2^−10)` | `< 2^−10/4 = 2.44·10⁻⁴` |
| output truncation to `1/R` grid (R6) | — | `< 2^−16 = 1.53·10⁻⁵` |
| clamp at `|z| = 10` (R5) | — | `< 4.54·10⁻⁵` |

**Total: `max|Δp| < 2.66·10⁻⁴`.** A label can differ from the float model's
only if `|p − 0.5| < 2.66·10⁻⁴`.

Measured (test sets, `scripts/prepare_real_datasets.py`):

| dataset | samples | max\|Δp\| | label agreement | accuracy delta | AUC delta |
|---|---|---|---|---|---|
| WBC | 205 | 0.000230 | 100% | 0 | −0.000000 |
| Pima | 231 | 0.000241 | 100% | 0 | −0.000082 |
| Credit | 9,000 | 0.000254 | 100% | 0 | −0.000004 |

All measured maxima sit below the analytic bound. The Go pipeline (real
proofs) reproduces the simulated accuracy exactly (WBC: 95.61% both).

## 4. Empirical tamper matrix

Every row is a test that feeds the constraint system a witness with exactly
one component tampered and asserts rejection.

| tampered component | circuit | test | result |
|---|---|---|---|
| output Y | single | `TestInvalidY` | rejected |
| lookup index ZTable | single | `TestInvalidZ` | rejected |
| remainder Rem (+1) | single | `TestWrongRemRejected` | rejected |
| remainder ≥ 2^22 (rebalanced) | single | `TestWrongRemRejected` | rejected |
| commitment (arbitrary) | single | `TestWrongCommitmentRejected` | rejected |
| commitment (other model's) | single | `TestModelSwapRejected` | rejected |
| feature x = 4096 (13 bits) | single | `TestXBoundary12Bit` | rejected |
| one Y of a batch | batch-prob | `TestBatchTamperedYRejected` | rejected |
| batch commitment | batch-prob | `TestBatchWrongCommitmentRejected` | rejected |
| one label bit of a batch | batch-label | `TestBatchLabelFlippedRejected` | rejected |
| one label bit (B×C) | one-vs-rest | `TestOneVsRestFlippedLabelRejected` | rejected |
| classifier order swap | one-vs-rest | `TestOneVsRestClassSwapRejected` | rejected |
| serialized proof byte-flip | verifier CLI / EVM | e2e + `test_RejectCorruptedProof` | rejected |
| x = 4095 (12-bit max) | single | `TestXBoundary12Bit` | **accepted** (boundary) |
| z = ±50 (clamp branches) | single | `TestClampSaturation` | **accepted**, saturated y |

Run: `go test ./... -count=1`.

## References

- [GWC19] Gabizon, Williamson, Ciobotaru. *PLONK*. ePrint 2019/953.
- [Hab22] Haböck. *Multivariate lookups based on logarithmic derivatives*. ePrint 2022/1530.
- Tramèr et al. *Stealing Machine Learning Models via Prediction APIs*. USENIX Security 2016.
