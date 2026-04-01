#!/bin/bash
# Runs ZKP batch predict on the 5-feature test dataset with a given batch size
# Usage: ./run_5f_zkp.sh <batch_size> <workers>

BATCH=${1:-20}
WORKERS=${2:-3}
WEIGHTS="-0.04232695012426456,0.2289242246189388,-0.040603009230595656,0.04991542646456064,0.028145240402081887"
BIAS="-4.530066986279241"
DATASET="data/synthetic_5f_test.csv"

echo "=============================================="
echo "  ZKP Run: 5-feature | batch=$BATCH | workers=$WORKERS"
echo "=============================================="

go run ./cmd/batch_predict/main.go \
  -dataset="$DATASET" \
  -weights="$WEIGHTS" \
  -bias="$BIAS" \
  -batch="$BATCH" \
  -workers="$WORKERS"
