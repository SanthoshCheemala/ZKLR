#!/bin/bash
# ================================================================
# HPC Comprehensive Grid Sweep Benchmark
# Tests multiple batch sizes and worker counts automatically.
# Usage:
#   chmod +x run_hpc_sweep.sh
#   ./run_hpc_sweep.sh
# ================================================================

set -e

WEIGHTS="0.0248928757,0.0229687007,0.0247936020,0.0249599228"
BIAS="-4.8863934725"
DATASET="data/test_3000.csv"

echo "=================================================="
echo "  ZKLR HPC Grid Sweep Benchmark (3000 samples)"
echo "=================================================="

# Define grids to sweep
BATCH_SIZES=(10 20 40 80)
WORKER_COUNTS=(1 4 8 16 32 64 96)

# Create a clean log for the summary
SUMMARY_FILE="results/hpc_grid_sweep_summary.txt"
echo "HPC Grid Sweep Summary" > "$SUMMARY_FILE"
echo "------------------------------------------------" >> "$SUMMARY_FILE"

for B in "${BATCH_SIZES[@]}"; do
    for W in "${WORKER_COUNTS[@]}"; do
        
        echo "[Running] Batch: $B | Workers: $W"
        
        # Output file name for this specific run
        OUT_FILE="results/grid_b${B}_w${W}.txt"
        
        # Run the generic batch predict code but ONLY capture the "[5] Summary" to avoid massive text output
        go run ./cmd/batch_predict/main.go \
          -dataset="$DATASET" \
          -weights="$WEIGHTS" \
          -bias="$BIAS" \
          -batch="$B" \
          -workers="$W" \
          2>&1 > "$OUT_FILE"
        
        # Extract the wall clock line to print immediately
        WALL_CLOCK=$(grep "Wall-clock/sample:" "$OUT_FILE" | awk '{print $2}')
        
        echo "   -> Wall-clock/sample: $WALL_CLOCK"
        echo "B=${B}, W=${W}, Wall=${WALL_CLOCK}" >> "$SUMMARY_FILE"
        
    done
done

echo "=================================================="
echo " Sweep Complete! Summary saved to $SUMMARY_FILE"
echo " Push the summary to GitHub: "
echo "   git add results/hpc_grid_sweep_summary.txt"
echo "   git commit -m \"Add comprehensive HPC sweep results\""
echo "   git push"
echo "=================================================="
