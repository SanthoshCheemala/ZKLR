#!/bin/bash
# ================================================================
# HPC Benchmark Script for ZK-LR 3K Dataset
# Run this on the HPC cluster after cloning the repo.
# Results saved to results/ and can be pulled back for the report.
#
# Usage:
#   chmod +x run_hpc_benchmark.sh
#   ./run_hpc_benchmark.sh
# ================================================================

set -e

WEIGHTS="0.0248928757,0.0229687007,0.0247936020,0.0249599228"
BIAS="-4.8863934725"
DATASET="data/test_3000.csv"

echo "=================================================="
echo "  ZK-LR HPC Benchmark — 3K Dataset, 4 Features"
echo "=================================================="
echo ""

# Detect CPU count
CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $CPUS CPU cores"
echo ""

# ── Run 1: batch=40, workers=6 (throughput-optimised) ─────────
echo "[1/3] Running batch=40, workers=6..."
go run ./cmd/batch_predict/main.go \
  -dataset="$DATASET" \
  -weights="$WEIGHTS" \
  -bias="$BIAS" \
  -batch="40" \
  -workers="6" \
  2>&1 | tee results/benchmark_3k_b40_w6.txt
echo ""

# ── Run 2: batch=20, workers=6 ────────────────────────────────
echo "[2/3] Running batch=20, workers=6..."
go run ./cmd/batch_predict/main.go \
  -dataset="$DATASET" \
  -weights="$WEIGHTS" \
  -bias="$BIAS" \
  -batch="20" \
  -workers="6" \
  2>&1 | tee results/benchmark_3k_b20_w6.txt
echo ""

# ── Run 3: batch=40, workers=auto (all cores) ─────────────────
echo "[3/3] Running batch=40, workers=$CPUS (all cores)..."
go run ./cmd/batch_predict/main.go \
  -dataset="$DATASET" \
  -weights="$WEIGHTS" \
  -bias="$BIAS" \
  -batch="40" \
  -workers="$CPUS" \
  2>&1 | tee results/benchmark_3k_b40_wall.txt
echo ""

echo "=================================================="
echo "  All benchmark runs complete!"
echo "  Output files:"
echo "    results/benchmark_3k_b40_w6.txt"
echo "    results/benchmark_3k_b20_w6.txt"
echo "    results/benchmark_3k_b40_wall.txt"
echo "=================================================="
