#!/bin/bash
# On-chain gas measurement for ZKLR proof verification (Phase 2 / Table T5).
#
# Regenerates proofs for a given batch size, exports the Solidity verifier and
# real-proof fixtures, and runs Foundry's gas report.
#
# Usage: ./scripts/run_gas_benchmark.sh [batch_size] [dataset] [weights] [bias]
set -euo pipefail

BATCH=${1:-5}
DATASET=${2:-data/test_10.csv}
WEIGHTS=${3:-"0.024892875697733793,0.02296870068086003,0.024793602045975024,0.024959922786682482"}
BIAS=${4:--4.886393472457494}

if ! command -v forge >/dev/null 2>&1; then
  echo "ERROR: Foundry not found. Install with:"
  echo "  curl -L https://foundry.paradigm.xyz | bash && foundryup"
  exit 1
fi

echo "=== [1/4] Proving (batch=$BATCH, label mode) ==="
go run ./cmd/batch_predict \
  -dataset="$DATASET" -batch="$BATCH" -mode=label \
  -weights="$WEIGHTS" -bias="$BIAS" \
  -export-proofs=results/proofs_gas

echo "=== [2/4] Exporting Solidity verifier + fixtures ==="
go run ./cmd/evm_export -dir=results/proofs_gas -out=contracts -max-fixtures=2

echo "=== [3/4] Installing forge-std (if missing) ==="
cd contracts
if [ ! -d lib/forge-std ]; then
  forge install foundry-rs/forge-std
fi

echo "=== [4/4] forge test --gas-report ==="
forge test --gas-report -vv

echo ""
echo "Gas of interest: ZKLRBatchVerifier.verifyBatch in the report above."
echo "Calldata size: proof (bytes) + 32 * public_inputs — printed by evm_export."
