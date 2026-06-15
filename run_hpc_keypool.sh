#!/bin/bash
# ================================================================
# ZKLR Key-Pool Memory/Throughput Sweep (Phase 3.8).
#
# Measures peak resident memory and wall-clock/sample across key-pool sizes at
# a FIXED worker count, exposing the memory<->throughput tradeoff that is one of
# ZKLR's original systems contributions.
#
#     chmod +x run_hpc_keypool.sh
#     ./run_hpc_keypool.sh
#
# Output: results/keypool_sweep.csv  (keys, peak_rss_mb, wall_per_sample, ...)
# Needs GNU time (/usr/bin/time -v) for peak RSS; falls back to no-RSS if absent.
#
# Override via env:
#   WORKERS=16 BATCH=80 KEY_SIZES="1 2 4 8 16" ./run_hpc_keypool.sh
# ================================================================
set -euo pipefail

DATASET="${DATASET:-data/credit_test.csv}"
WEIGHTS_FILE="${WEIGHTS_FILE:-data/credit_weights.txt}"
MODE="${MODE:-label}"
BATCH="${BATCH:-80}"
WORKERS="${WORKERS:-16}"
read -r -a KEY_SIZES <<< "${KEY_SIZES:-1 2 4 8 16}"
REPS="${REPS:-3}"
CSV="${CSV:-results/keypool_sweep.csv}"

WEIGHTS=$(grep -i '^Weights:' "$WEIGHTS_FILE" | cut -d: -f2 | tr -d ' ')
BIAS=$(grep -i '^Bias:' "$WEIGHTS_FILE" | cut -d: -f2 | tr -d ' ')
CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [[ "$WORKERS" -gt "$CPUS" ]]; then WORKERS="$CPUS"; fi

echo "=================================================="
echo "  ZKLR Key-Pool Memory Sweep"
echo "  batch=$BATCH workers=$WORKERS keys=[${KEY_SIZES[*]}] reps=$REPS"
echo "  (peak RSS self-reported by batch_predict via /proc/self/status VmHWM)"
echo "=================================================="

mkdir -p results bin
go build -o bin/batch_predict ./cmd/batch_predict

# Warm keys once.
./bin/batch_predict -dataset="$DATASET" -weights="$WEIGHTS" -bias="$BIAS" \
  -batch="$BATCH" -workers="$WORKERS" -mode="$MODE" >/dev/null 2>&1 || true

echo "keys,workers,batch,rep,peak_rss_mb,wall_per_sample,prove_per_sample" > "$CSV"
SWEEP_CSV="results/keypool_runs.csv"; rm -f "$SWEEP_CSV"

for K in "${KEY_SIZES[@]}"; do
  for ((run=1; run<=REPS; run++)); do
    printf "[keys=%-2s rep=%-2s] " "$K" "$run"
    ./bin/batch_predict -dataset="$DATASET" -weights="$WEIGHTS" -bias="$BIAS" \
      -batch="$BATCH" -workers="$WORKERS" -keys="$K" -mode="$MODE" \
      -csv="$SWEEP_CSV" -run="$run" >/dev/null 2>&1 || true
    # batch_predict's own CSV: prove_per_sample=$12, wall_per_sample=$13, peak_rss_mb=$16
    WALL=$(tail -1 "$SWEEP_CSV" | awk -F, '{print $13}')
    PROVE=$(tail -1 "$SWEEP_CSV" | awk -F, '{print $12}')
    RSS_MB=$(tail -1 "$SWEEP_CSV" | awk -F, '{print $16}')
    echo "${K},${WORKERS},${BATCH},${run},${RSS_MB},${WALL},${PROVE}" >> "$CSV"
    echo "peak_rss=${RSS_MB:-NA}MB wall/sample=${WALL}s"
  done
done

echo "=================================================="
echo " Key-pool sweep complete → $CSV"
echo " (plot peak_rss_mb and wall_per_sample vs keys for the tradeoff curve)"
echo "=================================================="
