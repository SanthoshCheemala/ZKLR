# ZKLR

Zero-Knowledge Logistic Regression pipeline built with gnark (PLONK, BN254).

ZKLR proves that ML predictions were computed by a specific, pre-announced
model — without revealing the model weights. Each proof is bound to a public
**model commitment** `C = MiMC(W..., B)`, and by default only the predicted
**class label** is revealed per sample (never the raw probability, which would
allow model extraction).

## What a proof states

> "For the weights committed to in C, the committed model predicts label L_i
> for public features X_i — for all i in this batch."

- Private: weights W, bias B (plus internal lookup witnesses)
- Public: features X, labels L (or probabilities Y in `-mode=prob`), commitment C

## Project Structure

- `cmd/batch_predict` — prediction + proving pipeline (batched, parallel)
- `cmd/verify` — independent verifier: checks exported proofs from files alone
- `cmd/export` — verification key export for the smart-contract workflow
- `cmd/evm_export` — Solidity verifier + real-proof fixtures for gas benchmarks
- `cmd/groth16_baseline` — same circuit on Groth16/R1CS for the comparison table
- `cmd/constraint_scan` — constraints vs batch size (cost-model input)
- `prover` — setup, proving, verification, key/SRS I/O, artifact export
- `circuit` — gnark circuits (single, batch, batch-label, one-vs-rest
  multi-class) and tests
- `contracts` — Foundry project: PLONK verifier + commitment-pinning wrapper + tests
- `data` — datasets and model weights (incl. WBC / Pima / Taiwan-credit, prepared
  by `scripts/prepare_real_datasets.py`)
- `results` — generated outputs, fidelity reports, cached setup keys
- `docs` — architecture, trusted setup, integration notes, journal plan

## Real-Dataset Fidelity Study

```bash
python3 scripts/prepare_real_datasets.py   # downloads, quantizes, trains, reports
```

Produces ZKLR-format train/test CSVs, weights files (with ready-to-run
commands), and `results/fidelity_*.txt` showing float-vs-circuit accuracy,
label agreement, |Δp| and AUC delta per dataset.

## On-Chain Gas Benchmark

```bash
./scripts/run_gas_benchmark.sh 20    # needs Foundry (forge)
```

Proves a batch, exports `contracts/src/PlonkVerifier.sol` + fixtures from the
real proofs, and runs `forge test --gas-report` (valid, tampered-label,
wrong-commitment and corrupted-proof cases).

## Build & Test

```bash
go build ./...
go test ./...        # includes the negative-test suite (tamper matrix)
```

## Run Batch Prediction

Weights are required (one per feature, comma-separated):

```bash
go run ./cmd/batch_predict \
  -dataset=data/test_200.csv \
  -weights="0.0249,0.0230,0.0248,0.0250" \
  -bias=-4.886
```

Key flags:

- `-mode=label` (default) — only the class bit is public per sample.
  `-mode=prob` exposes exact probabilities: **observers can then reconstruct
  the weights from ~d+1 predictions**; use only with trusted verifiers.
- `-batch=0` — auto batch size from CPU cores (80 on HPC, 40/20/10 below)
- `-workers=0` — auto (50% of cores); `-keys=0` — auto key-pool size
- `-srs=<file>` — ceremony SRS for production keys (default: DEV-ONLY local
  SRS; see [docs/TRUSTED_SETUP.md](docs/TRUSTED_SETUP.md))
- `-export-proofs=<dir>` — write vk + proofs + public witnesses for
  independent verification

The pipeline prints the model commitment at startup — publish it once;
verifiers check every proof against it.

## Independent Verification

```bash
go run ./cmd/batch_predict ... -export-proofs=results/proofs
go run ./cmd/verify -dir=results/proofs
```

`cmd/verify` shares no state with the prover: it loads the verification key,
proofs, and public witnesses (features, labels, commitment) from disk.
Single-file mode: `go run ./cmd/verify -vk=... -proof=... -public=...`

## Export Verification Key for Smart Contract Flow

```bash
go run ./cmd/export -batch=80 -features=4 -mode=label -out=results/Verifier
gnark export-solidity -vk=results/Verifier.vk -o contracts/ -p Verifier
```

Production contracts must use keys from a ceremony SRS (`-srs`), never the
dev default — see [docs/TRUSTED_SETUP.md](docs/TRUSTED_SETUP.md).

## Notes

- Setup keys are cached in `results/` (keyed by batch size, features, mode;
  `ceremony_` prefix for real-SRS keys) to accelerate repeated runs.
- Proof size is constant per batch (the pipeline reports the measured size).
- Completeness domain: features must fit 12 bits (0–4095) and the model output
  z = W·X + B must satisfy |z| < 1000; |z| > 10 saturates to the clamped
  sigmoid value (which cannot flip a label).

## Reproduce All Reported Numbers

```bash
make reproduce   # datasets + fidelity + cost model + Groth16 + EZKL baselines
make test        # full suite incl. the tamper matrix
make gas         # on-chain gas (requires Foundry)
```

## Documentation

- [docs/correctness.md](docs/correctness.md) — the proven relation, gadget
  soundness, fidelity bound, tamper matrix
- [docs/comparative_analysis.md](docs/comparative_analysis.md) — measured
  head-to-head vs Groth16 and EZKL + related-work matrix
- [docs/circuit_architecture.md](docs/circuit_architecture.md)
- [docs/TRUSTED_SETUP.md](docs/TRUSTED_SETUP.md)
- [docs/SMART_CONTRACT_INTEGRATION.md](docs/SMART_CONTRACT_INTEGRATION.md)
- [docs/JOURNAL_SUBMISSION_PLAN.md](docs/JOURNAL_SUBMISSION_PLAN.md)
