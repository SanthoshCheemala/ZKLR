// cmd/benchmark/main.go — Configuration sweep benchmark.
//
// Tests multiple batch sizes and worker counts to find the optimal config.
// Results are saved to results/config_sweep.txt.
//
// Usage:
//
//	go run ./cmd/benchmark/
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/santhoshcheemala/ZKLR/prover"
)

const (
	W1_FLOAT      = -3.3144933046
	W2_FLOAT      = 0.3877500778
	B_FLOAT       = 281.2861173099
	DATASET       = "data/test_1k.csv"
	TOTAL_SAMPLES = 1000
)

type BenchResult struct {
	BatchSize   int
	Workers     int
	SetupTime   time.Duration
	PredictTime time.Duration
	VerifyTime  time.Duration
	Constraints int
	Correct     int
}

func main() {
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  ZK-LR Configuration Sweep Benchmark")
	fmt.Println("═══════════════════════════════════════════════════════")

	testFeatures := loadTestFeatures()
	fmt.Printf("  Dataset: %s (%d samples)\n\n", DATASET, len(testFeatures))

	// Configurations to test
	batchSizes := []int{5, 10, 20, 50, 100}
	workerCounts := []int{1} // mutex serializes proves, so workers=1 is optimal

	var results []BenchResult

	for _, batchSize := range batchSizes {
		for _, workers := range workerCounts {
			fmt.Printf("─── batch=%d, workers=%d ───────────────────────\n", batchSize, workers)

			// Setup
			setupStart := time.Now()
			// Assuming benchmark data uses 2 features for backwards compatibility default
			setup, err := prover.RunBatchSetup(batchSize, 2)
			setupDone := time.Since(setupStart)
			if err != nil {
				fmt.Printf("  Setup FAILED: %v\n", err)
				continue
			}
			fmt.Printf("  Setup:       %v (%d constraints)\n", setupDone.Round(time.Millisecond), setup.NumConstraints)

			// Run predictions
			predStart := time.Now()
			batchResults := prover.BatchPredictParallel(setup, []float64{W1_FLOAT, W2_FLOAT}, B_FLOAT, testFeatures, workers)
			predDone := time.Since(predStart)

			// Count results
			correct := 0
			var totalVerify time.Duration
			failed := 0
			for _, br := range batchResults {
				if br == nil || br.Error != nil {
					failed++
					continue
				}
				totalVerify += br.VerifyTime
			}

			if failed > 0 {
				fmt.Printf("  FAILED batches: %d\n", failed)
			}

			perSample := predDone.Seconds() / float64(len(testFeatures))
			speedup := 2.0 / perSample

			fmt.Printf("  Predict:     %v  (%.3fs/sample, %.1fx speedup)\n",
				predDone.Round(time.Millisecond), perSample, speedup)
			fmt.Printf("  Verify:      %v\n", totalVerify.Round(time.Millisecond))
			fmt.Printf("  Batches:     %d × %d samples\n", len(batchResults), batchSize)
			fmt.Println()

			results = append(results, BenchResult{
				BatchSize:   batchSize,
				Workers:     workers,
				SetupTime:   setupDone,
				PredictTime: predDone,
				VerifyTime:  totalVerify,
				Constraints: setup.NumConstraints,
				Correct:     correct,
			})
		}
	}

	// Print summary table
	fmt.Println("\n═══════════════════════════════════════════════════════")
	fmt.Println("  RESULTS SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("  %-10s %-10s %-15s %-15s %-12s %-10s\n",
		"Batch", "Workers", "Constraints", "Predict Time", "Per Sample", "Speedup")
	fmt.Println("  " + strings.Repeat("─", 72))

	bestIdx := -1
	var bestTime time.Duration
	for i, r := range results {
		perSample := r.PredictTime.Seconds() / float64(TOTAL_SAMPLES)
		speedup := 2.0 / perSample
		marker := ""
		if bestIdx == -1 || r.PredictTime < bestTime {
			bestTime = r.PredictTime
			bestIdx = i
		}
		fmt.Printf("  %-10d %-10d %-15d %-15v %-12s %-10s%s\n",
			r.BatchSize, r.Workers, r.Constraints,
			r.PredictTime.Round(time.Second),
			fmt.Sprintf("%.3fs", perSample),
			fmt.Sprintf("%.1fx", speedup),
			marker,
		)
	}
	if bestIdx >= 0 {
		best := results[bestIdx]
		perSample := best.PredictTime.Seconds() / float64(TOTAL_SAMPLES)
		fmt.Printf("\n  ⭐ Optimal: batch=%d, workers=%d → %.3fs/sample\n",
			best.BatchSize, best.Workers, perSample)
	}

	// Save results
	saveResults(results)
	fmt.Println("\n  Results saved → results/config_sweep.txt")
	fmt.Println("✅ Done!")
}

func loadTestFeatures() [][]int {
	f, err := os.Open(DATASET)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open %s: %v\n", DATASET, err)
		os.Exit(1)
	}
	defer f.Close()

	var features [][]int
	first := true
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if first {
			first = false
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) >= 2 {
			lastIdx := len(parts) - 1
			row := make([]int, lastIdx)
			for j := 0; j < lastIdx; j++ {
				row[j], _ = strconv.Atoi(parts[j])
			}
			features = append(features, row)
		}
	}
	return features
}

func saveResults(results []BenchResult) {
	os.MkdirAll("results", 0o755)
	f, err := os.Create("results/config_sweep.txt")
	if err != nil {
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "ZK-LR Configuration Sweep Results\n")
	fmt.Fprintf(f, "==================================\n")
	fmt.Fprintf(f, "Dataset: %s (%d samples)\n\n", DATASET, TOTAL_SAMPLES)
	fmt.Fprintf(f, "%-10s %-10s %-15s %-12s %-12s %-10s\n",
		"Batch", "Workers", "Constraints", "Setup", "Predict", "Per Sample")
	fmt.Fprintf(f, "%s\n", strings.Repeat("─", 70))

	var best *BenchResult
	for i := range results {
		r := &results[i]
		if best == nil || r.PredictTime < best.PredictTime {
			best = r
		}
		perSample := r.PredictTime.Seconds() / float64(TOTAL_SAMPLES)
		fmt.Fprintf(f, "%-10d %-10d %-15d %-12v %-12v %.3fs\n",
			r.BatchSize, r.Workers, r.Constraints,
			r.SetupTime.Round(time.Millisecond),
			r.PredictTime.Round(time.Second),
			perSample,
		)
	}

	if best != nil {
		fmt.Fprintf(f, "\nOptimal: batch=%d, workers=%d → %.3fs/sample (%.1fx speedup vs sequential)\n",
			best.BatchSize, best.Workers,
			best.PredictTime.Seconds()/float64(TOTAL_SAMPLES),
			2.0/(best.PredictTime.Seconds()/float64(TOTAL_SAMPLES)))
	}
}
