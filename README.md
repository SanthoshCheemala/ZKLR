# ZKLR

Zero-Knowledge Logistic Regression pipeline built with gnark (PLONK, BN254).

This project generates and verifies proofs for ML predictions while keeping model weights private.

## Current Defaults

The batch prediction command is currently configured with benchmarked defaults:

- Batch size: 80
- Workers: auto (50% of available CPU cores)
- Key pool: auto
- Features: 4
- Default dataset: data/test_200.csv

## Project Structure

- cmd/batch_predict: Main prediction and proof pipeline
- cmd/export: Verification key export helper for smart contract workflow
- cmd/worker_benchmark: Minimal placeholder command
- prover: Setup, proving, verification, key I/O, export utilities
- circuit: gnark circuit definitions and tests
- data: datasets and model weights
- results: generated outputs and cached setup keys
- docs: architecture and integration notes

## Prerequisites

- Go 1.21+ (recommended)
- gnark-compatible environment

Optional for Solidity integration:

- gnark CLI
- Truffle or Hardhat

## Build

```bash
go build ./...
```

## Test

```bash
go test ./...
```

## Run Batch Prediction

Use default config:

```bash
go run ./cmd/batch_predict
```

Run with dataset override:

```bash
go run ./cmd/batch_predict -dataset=data/synthetic_4f_test.csv
```

Important flags:

- -workers=0 uses auto mode (half CPU cores)
- -keys=0 uses auto key pool sizing

## Export Verification Key for Smart Contract Flow

Generate a verification key file:

```bash
go run ./cmd/export -out=results/Verifier
```

This creates:

- results/Verifier.vk

Then generate Solidity verifier contract with gnark CLI:

```bash
gnark export-solidity -vk=results/Verifier.vk -o contracts/ -p Verifier
```

## Output Files

Main output files produced by the pipeline:

- results/batch_prediction_results.txt
- results/batch_pk_b80_f4.key
- results/batch_vk_b80_f4.key

## Notes

- Setup keys are cached in results to accelerate repeated runs.
- Proof size is constant per batch (as reported by the pipeline output).
- Model weights remain private in prover flow; only proof and public inputs are used for on-chain verification.

## Documentation

- docs/circuit_architecture.md
- docs/SMART_CONTRACT_INTEGRATION.md
- docs/RESTRUCTURE_PLAN.md
