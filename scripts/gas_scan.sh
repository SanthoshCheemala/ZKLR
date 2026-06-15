#!/bin/bash
# ================================================================
# On-chain verification gas vs batch size (case-study Table T5).
#
# For each batch size, regenerates a real credit-model proof, exports the
# matching Solidity verifier + fixtures, and records forge's verifyBatch gas.
# This exposes the prove-vs-verify tradeoff: proving favours large batches
# (FFT-domain fill), but on-chain verify gas grows with the public-input count
# (B*F + B + 1), so per-decision gas has its own optimum.
#
#   ./scripts/gas_scan.sh
#
# Output: results/gas_scan.csv  (batch, public_inputs, verify_gas, deploy_gas)
# Needs: forge on PATH, bin/batch_predict + bin/evm_export built.
# ================================================================
set -euo pipefail
export PATH="$HOME/.foundry/bin:$PATH"

DATASET="${DATASET:-data/credit_test.csv}"
WEIGHTS_FILE="${WEIGHTS_FILE:-data/credit_weights.txt}"
read -r -a BATCHES <<< "${BATCHES:-1 5 10 20 40}"
CSV="${CSV:-results/gas_scan.csv}"

WEIGHTS=$(grep -i '^Weights:' "$WEIGHTS_FILE" | cut -d: -f2 | tr -d ' ')
BIAS=$(grep -i '^Bias:' "$WEIGHTS_FILE" | cut -d: -f2 | tr -d ' ')
FEATURES=$(head -1 "$DATASET" | awk -F, '{print NF-1}')

command -v forge >/dev/null || { echo "forge not on PATH"; exit 1; }
[ -x bin/batch_predict ] || go build -o bin/batch_predict ./cmd/batch_predict
[ -x bin/evm_export ]    || go build -o bin/evm_export ./cmd/evm_export

echo "batch,features,public_inputs,verify_gas,deploy_gas" > "$CSV"
mkdir -p results

for B in "${BATCHES[@]}"; do
  rows=$((2 * B))                       # two batches → two fixtures
  head -n $((rows + 1)) "$DATASET" > data/_gas_subset.csv
  printf "[gas] batch=%-3s proving... " "$B"
  ./bin/batch_predict -dataset=data/_gas_subset.csv -batch="$B" -mode=label \
    -weights="$WEIGHTS" -bias="$BIAS" -export-proofs=results/_gas_proofs >/dev/null 2>&1
  ./bin/evm_export -dir=results/_gas_proofs -out=contracts -max-fixtures=1 >/dev/null 2>&1

  report=$(cd contracts && forge test --gas-report -vv 2>/dev/null)
  vgas=$(printf '%s\n' "$report" | grep 'test_Gas_VerifyBatch' | grep -oE 'gas: [0-9]+' | grep -oE '[0-9]+' | head -1)
  dgas=$(printf '%s\n' "$report" | awk '/Deployment Cost/{getline; gsub(/[^0-9]/,"",$0); print; exit}')
  pubin=$((B * FEATURES + B + 1))
  echo "${B},${FEATURES},${pubin},${vgas:-NA},${dgas:-NA}" >> "$CSV"
  echo "public_inputs=${pubin} verify_gas=${vgas:-NA} (per-decision $(( ${vgas:-0} / B )))"
done

rm -f data/_gas_subset.csv
echo "=================================================="
echo " Gas scan complete → $CSV"
column -s, -t "$CSV"
echo "=================================================="
