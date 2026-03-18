#!/bin/bash
# HPC Benchmark Script for ZKLR
# Generates datasets, trains model, and runs benchmarks
# All-in-one script for HPC deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUTPUT_DIR="results/hpc_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Test configurations
TRAIN_SIZE=10000
TEST_SIZES="1000,2000,3000"
WORKER_COUNTS=(1 2 4 8 16 32 64)
BATCH_SIZES=(10 20 40 80 100)

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  ZKLR HPC Benchmark - Full Pipeline"
echo "=============================================="
echo "  Project dir: $PROJECT_DIR"
echo "  Train size:  $TRAIN_SIZE"
echo "  Test sizes:  $TEST_SIZES"
echo "=============================================="

# ─── Step 1: Generate datasets and train model ───
echo ""
echo "[1/3] Generating datasets and training model..."
python3 scripts/prepare_4f_model.py --train $TRAIN_SIZE --test $TEST_SIZES

# ─── Step 2: Read weights from file ───
echo ""
echo "[2/3] Loading model weights..."
WEIGHTS_FILE="data/model_weights.txt"
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ERROR: Weights file not found: $WEIGHTS_FILE"
    exit 1
fi

WEIGHTS=$(grep "Weights:" "$WEIGHTS_FILE" | cut -d' ' -f2)
BIAS=$(grep "Bias:" "$WEIGHTS_FILE" | cut -d' ' -f2)

echo "  Weights: $WEIGHTS"
echo "  Bias: $BIAS"

# ─── Step 3: Run benchmarks ───
echo ""
echo "[3/3] Running benchmarks..."

# Get system info
TOTAL_CORES=$(nproc)
CPU_MODEL=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "Unknown")
TOTAL_MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "Unknown")

# Initialize output files
CSV_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.csv"
JSON_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.json"

echo "dataset,workers,batch_size,samples,constraints,setup_time_s,predict_time_s,prove_time_s,verify_time_s,per_sample_s,accuracy,speedup" > "$CSV_FILE"

cat > "$JSON_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "system": "$(uname -n)",
  "cpu": "$CPU_MODEL",
  "total_cores": $TOTAL_CORES,
  "total_memory_gb": "$TOTAL_MEM",
  "train_size": $TRAIN_SIZE,
  "test_sizes": [$TEST_SIZES],
  "worker_counts": [$(IFS=,; echo "${WORKER_COUNTS[*]}")],
  "batch_sizes": [$(IFS=,; echo "${BATCH_SIZES[*]}")],
  "weights": "$WEIGHTS",
  "bias": $BIAS,
  "results": [
EOF

echo ""
echo "System: $CPU_MODEL"
echo "Cores: $TOTAL_CORES | Memory: ${TOTAL_MEM}GB"
echo "Workers: ${WORKER_COUNTS[*]}"
echo "Batches: ${BATCH_SIZES[*]}"
echo ""

FIRST=true
RUN_COUNT=0

# Convert test sizes string to array
IFS=',' read -ra TEST_SIZE_ARR <<< "$TEST_SIZES"
TOTAL_RUNS=$((${#TEST_SIZE_ARR[@]} * ${#WORKER_COUNTS[@]} * ${#BATCH_SIZES[@]}))

echo "Total benchmark runs: $TOTAL_RUNS"
echo "─────────────────────────────────────────────"

for TEST_SIZE in "${TEST_SIZE_ARR[@]}"; do
    DATASET="data/test_${TEST_SIZE}.csv"
    
    if [ ! -f "$DATASET" ]; then
        echo "WARNING: Dataset not found: $DATASET, skipping..."
        continue
    fi
    
    echo ""
    echo ">>> Dataset: $DATASET ($TEST_SIZE samples)"
    
    for BATCH in "${BATCH_SIZES[@]}"; do
        for WORKERS in "${WORKER_COUNTS[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))
            echo -n "  [$RUN_COUNT/$TOTAL_RUNS] workers=$WORKERS, batch=$BATCH ... "
            
            # Run benchmark
            OUTPUT=$(go run ./cmd/batch_predict/ \
                -workers=$WORKERS \
                -batch=$BATCH \
                -dataset="$DATASET" \
                -weights="$WEIGHTS" \
                -bias="$BIAS" 2>&1) || true
            
            # Parse results
            CONSTRAINTS=$(echo "$OUTPUT" | grep "Constraints:" | awk '{print $2}' || echo "0")
            SAMPLES=$(echo "$OUTPUT" | grep "Samples:" | head -1 | awk '{print $2}' || echo "0")
            SETUP_TIME=$(echo "$OUTPUT" | grep -E "^\s+Total:" | head -1 | awk '{print $2}' || echo "0s")
            PREDICT_TIME=$(echo "$OUTPUT" | grep "Total predict time:" | awk '{print $4}' || echo "0s")
            PROVE_TIME=$(echo "$OUTPUT" | grep "Total prove time:" | awk '{print $5}' || echo "0s")
            VERIFY_TIME=$(echo "$OUTPUT" | grep "Total verify time:" | awk '{print $4}' || echo "0s")
            PER_SAMPLE=$(echo "$OUTPUT" | grep "Wall-clock/sample:" | awk '{print $2}' | sed 's/s//' || echo "0")
            ACCURACY=$(echo "$OUTPUT" | grep "Accuracy:" | tail -1 | awk '{print $2}' | sed 's/%//' || echo "0")
            SPEEDUP=$(echo "$OUTPUT" | grep "Speedup vs single:" | awk '{print $4}' | sed 's/x//' || echo "0")
            
            # Time conversion function
            time_to_seconds() {
                local t=$1
                if [[ $t == *"m"* ]]; then
                    mins=$(echo $t | sed 's/m.*//')
                    rest=$(echo $t | sed 's/.*m//' | sed 's/s//')
                    echo "$mins * 60 + $rest" | bc 2>/dev/null || echo "0"
                elif [[ $t == *"ms"* ]]; then
                    echo $t | sed 's/ms//' | awk '{print $1/1000}'
                else
                    echo $t | sed 's/s//'
                fi
            }
            
            SETUP_S=$(time_to_seconds "$SETUP_TIME")
            PREDICT_S=$(time_to_seconds "$PREDICT_TIME")
            PROVE_S=$(time_to_seconds "$PROVE_TIME")
            VERIFY_S=$(time_to_seconds "$VERIFY_TIME")
            
            # Write CSV
            echo "$TEST_SIZE,$WORKERS,$BATCH,${SAMPLES:-0},${CONSTRAINTS:-0},$SETUP_S,$PREDICT_S,$PROVE_S,$VERIFY_S,${PER_SAMPLE:-0},${ACCURACY:-0},${SPEEDUP:-0}" >> "$CSV_FILE"
            
            # Write JSON
            if [ "$FIRST" = true ]; then
                FIRST=false
            else
                echo "," >> "$JSON_FILE"
            fi
            
            cat >> "$JSON_FILE" << EOF
    {
      "dataset_size": $TEST_SIZE,
      "workers": $WORKERS,
      "batch_size": $BATCH,
      "samples": ${SAMPLES:-0},
      "constraints": ${CONSTRAINTS:-0},
      "setup_time_s": $SETUP_S,
      "predict_time_s": $PREDICT_S,
      "prove_time_s": $PROVE_S,
      "verify_time_s": $VERIFY_S,
      "per_sample_s": ${PER_SAMPLE:-0},
      "accuracy": ${ACCURACY:-0},
      "speedup": ${SPEEDUP:-0}
    }
EOF
            
            echo "${PER_SAMPLE:-N/A}s/sample | ${ACCURACY:-N/A}% | ${SPEEDUP:-N/A}x"
        done
    done
done

echo "" >> "$JSON_FILE"
echo "  ]" >> "$JSON_FILE"
echo "}" >> "$JSON_FILE"

echo ""
echo "=============================================="
echo "  Benchmark Complete!"
echo "=============================================="
echo "  Total runs: $RUN_COUNT"
echo "  CSV: $CSV_FILE"
echo "  JSON: $JSON_FILE"
echo "=============================================="

# Print summary table
echo ""
echo "Summary - Best per-sample time for each dataset/worker:"
echo "Dataset | Workers | Best Time | Batch"
echo "--------|---------|-----------|------"
for TEST_SIZE in "${TEST_SIZE_ARR[@]}"; do
    for W in "${WORKER_COUNTS[@]}"; do
        BEST=$(grep "^$TEST_SIZE,$W," "$CSV_FILE" | sort -t',' -k10 -n | head -1)
        if [ -n "$BEST" ]; then
            B=$(echo $BEST | cut -d',' -f3)
            T=$(echo $BEST | cut -d',' -f10)
            echo " ${TEST_SIZE}   |    $W    |   ${T}s   |  $B"
        fi
    done
done
