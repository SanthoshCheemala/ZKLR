# ZKLR Comparative Analysis

## 1. Quantitative head-to-head (measured)

Same statement ("the private LR model evaluates these public WBC inputs"),
same model (9 features), same batch (B=20), same machine.

Hardware: Apple Silicon laptop (8 cores), macOS; Go 1.24.1, gnark v0.11.0,
EZKL 23.0.5 (Python bindings). Single proof, no parallelism. ZKLR/Groth16 use
dev SRS/setup; EZKL fetched its public KZG SRS (logrows 16). Numbers below
are from single representative runs on this hardware — re-measure with
N ≥ 10 repetitions on the target machine for publication (see plan §3.7).

| | **ZKLR (PLONK + lookup)** | **Groth16 (same circuit)** | **EZKL (halo2 + KZG)** |
|---|---|---|---|
| constraints | 330,617 (SCS) | 170,672 (R1CS) | logrows 16 (2^16 rows) |
| setup time | 7.5 s | 10.8 s | 2.8 s |
| prove / batch | ~29 s | 0.49 s | 2.53 s |
| prove / sample | 1.45 s | 0.025 s | 0.127 s |
| verify | 1.5 ms | 1.9 ms | 30 ms |
| proof size | **584 B** | **196 B** | 3,648 B |
| proving key | 33 MB | (in-memory) | **327 MB** |
| setup type | **universal** (one SRS for all circuits) | per-circuit MPC | **universal** (one SRS) |
| output privacy | **label-only mode** | label-only (same circuit) | outputs public (probabilities) |
| model binding | **in-circuit MiMC commitment** | same circuit | param hashing available |
| EVM verifier | **gnark export, 584 B calldata proof** | gnark export | EZKL export, larger calldata |

Sources: `results/ezkl_baseline_b20.txt`, `results/groth16_baseline_wbc_b20.txt`,
`results/plonk_b1_baseline_wbc.txt`, WBC pipeline run (b=20).

### Honest reading

- **Raw proving speed: ZKLR's PLONK pipeline loses, clearly.** Groth16 proves
  the identical circuit ~60× faster; EZKL ~11× faster. The gnark PLONK prover
  pays heavily for the lookup argument (BSB22 commitments) at this circuit size.
- **Where ZKLR wins:**
  - *Universal setup + small proofs together.* Groth16's 196 B proofs require
    a new multi-party ceremony for every batch size, feature count, and model
    architecture — operationally prohibitive for a model marketplace where
    circuits change. EZKL shares the universal-setup property but its proofs
    are 6.2× larger (3,648 B vs 584 B — calldata cost on-chain) and verify
    20× slower.
  - *Memory.* EZKL's 327 MB proving key vs ZKLR's 33 MB matters for the
    key-pool parallelism strategy (16 concurrent provers: 0.5 GB vs 5+ GB).
  - *Output privacy.* Label-only mode closes the Tramèr-style extraction
    channel; EZKL publishes the output tensor (probabilities) as-is.
  - *Throughput recovery via batching + parallelism.* The amortization model
    (107.7K + 11.1K·B constraints, R² = 1.0) plus the worker/key-pool
    pipeline brought wall-clock to 0.16 s/sample on HPC (28-config sweep) and
    0.42 s/sample on this laptop — the system-level answer to the
    per-proof speed gap.
- **Takeaway for positioning:** ZKLR is not "the fastest prover"; it is a
  *deployable end-to-end system* — committed model, universal setup,
  extraction-resistant outputs, constant 584 B on-chain proofs, measured gas —
  with a validated cost model for choosing batch size.

## 2. Qualitative comparison with related work

| system | model class | proof system | activation handling | batching/amortization | model binding | on-chain | code |
|---|---|---|---|---|---|---|---|
| **ZKLR (this work)** | LR (binary, OvR multi-class) | PLONK + logderiv lookup (BN254) | shared lookup table | **table shared across B samples × C classes; T = a + b·(B·C), R² ≈ 1** | MiMC commitment (in-circuit) | PLONK verifier + commitment pinning, gas-measured | Go/gnark, open |
| ZEN (ePrint 2021/087) | quantized NN | Groth16 (R1CS) | quantization-friendly encodings | per-inference | — | — | open |
| zkCNN (CCS '21) | CNN | GKR + polynomial commitment | piecewise/poly approx | per-inference (sumcheck efficiency) | — | — | open |
| Mystique (USENIX Sec '21) | NN | interactive ZK (VOLE) | conversions arith/bool | amortized via VOLE batching | — | — | open |
| ezDPS (PoPETS '23) | classical ML pipeline | Spartan | poly approx | per-stage | commitments to pipeline | — | open |
| zkml / Kang et al. | ImageNet-scale NN | halo2 | lookups | per-inference | — | — | open |
| **EZKL** | arbitrary ONNX | halo2 + KZG | lookups (auto) | one graph = one proof; batch via input tensor | param hashing option | EVM verifier export | open, production |
| zkLLM (CCS '24) | LLMs | specialized (tlookup/zkAttn) | custom lookup args | per-inference | — | — | open |

### Positioning paragraph (for the paper)

Prior ZKML systems either target much larger models with heavyweight
machinery (zkCNN, zkml, zkLLM) or provide general compilers without a
deployment story for *recurring, model-pinned* inference (EZKL proves what a
graph computed, but does not by default bind successive proofs to one
published model identity, expose only decision bits, or come with a
batch-size cost model). ZKLR's contribution is narrower and deeper: for the
widely deployed LR family, it demonstrates an end-to-end pipeline —
commitment-bound model, label-only outputs, universal SRS, parallel batched
proving with an exact amortization law validated in two dimensions (samples
and classes), independent file-based verification, and an EVM verifier with
commitment pinning — with every claim carried by a measurement or a test.

## 3. Reproduction

```bash
python3 scripts/prepare_real_datasets.py                  # data + fidelity (T4)
go test ./... -count=1                                     # tamper matrix (§4 of correctness.md)
go run ./cmd/constraint_scan -features=9 -out=...          # cost model input (F1)
python3 scripts/fit_cost_model.py                          # fit + figure
go run ./cmd/groth16_baseline -batch=20 ...                # Groth16 column
python3 scripts/ezkl_baseline.py --batch 20                # EZKL column
./scripts/run_gas_benchmark.sh 20                          # gas (needs Foundry)
```
