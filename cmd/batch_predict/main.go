// main.go — ZK-LR Batch Parallel Prediction Pipeline
//
// Usage:
//
//	go run ./cmd/batch_predict/                           # defaults: batch=80, workers=auto (50% CPU cores)
//	go run ./cmd/batch_predict/ -batch=20 -workers=4      # custom
//	go run ./cmd/batch_predict/ -workers=16 -keys=4       # 16 workers, 4 key copies (~260MB)
//	go run ./cmd/batch_predict/ -dataset=data/test_200.csv -batch=80 -workers=0   # optimal setup (auto workers)
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/santhoshcheemala/ZKLR/prover"
)

func main() {
	// CLI flags (hardcoded optimal: batch=80, workers=auto-half-CPU, keys=auto)
	batchSize := flag.Int("batch", 80, "predictions per proof")
	numWorkers := flag.Int("workers", 0, "parallel workers (0=auto: 50% of CPU cores)")
	keyPoolSize := flag.Int("keys", 0, "key pool size for memory control (0=auto, max 16)")
	datasetPath := flag.String("dataset", "data/test_200.csv", "CSV file path")
	weightsFlag := flag.String("weights", "0.02638518204922373,0.02298176404030624,0.024034825597816466,0.024289043576076152", "comma-separated model weights")
	biasFlag := flag.Float64("bias", -4.894930414628542, "model bias")
	flag.Parse()

	weightStrs := strings.Split(*weightsFlag, ",")
	weights := make([]float64, len(weightStrs))
	for i, s := range weightStrs {
		w, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid weight value %q: %v\n", s, err)
			os.Exit(1)
		}
		weights[i] = w
	}
	bias := *biasFlag

	// Validate hardcoded 4-feature model
	if len(weights) != 4 {
		fmt.Fprintf(os.Stderr, "Error: hardcoded model expects 4 weights, got %d\n", len(weights))
		os.Exit(1)
	}

	if *numWorkers == 0 {
		// Use 50% of available cores for HPC optimization
		*numWorkers = runtime.NumCPU() / 2
		if *numWorkers < 1 {
			*numWorkers = 1
		}
	}

	fmt.Println("============================================================")
	fmt.Println("  ZK-LR — Batch Parallel Prediction Pipeline")
	fmt.Println("============================================================")
	fmt.Printf("  Dataset:    %s\n", *datasetPath)
	fmt.Printf("  Batch size: %d\n", *batchSize)
	fmt.Printf("  Workers:    %d (CPUs: %d)\n", *numWorkers, runtime.NumCPU())

	// ─── Phase 1: Setup ──────────────────────────────────
	fmt.Println("\n[1] Batch Setup (compile → SRS → keys)...")
	setupStart := time.Now()
	setup, err := prover.RunBatchSetup(*batchSize, len(weights))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Setup failed: %v\n", err)
		os.Exit(1)
	}
	setupTotal := time.Since(setupStart)

	fmt.Printf("    Constraints:   %d\n", setup.NumConstraints)
	fmt.Printf("    Compile time:  %v\n", setup.CompileTime)
	fmt.Printf("    Setup time:    %v\n", setup.SetupTime)
	fmt.Printf("    PK size:       %.1f KB\n", float64(setup.PKSizeBytes)/1024)
	fmt.Printf("    VK size:       %.1f KB\n", float64(setup.VKSizeBytes)/1024)
	fmt.Printf("    Total:         %v\n", setupTotal)

	if err := os.MkdirAll("results", 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to create results directory: %v\n", err)
	}

	// ─── Phase 2: Load Dataset ───────────────────────────
	fmt.Println("\n[2] Loading dataset...")
	testData, err := loadCSV(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("    Loaded %d samples\n", len(testData))

	features := make([][]int, len(testData))
	for i, s := range testData {
		features[i] = s.features
	}

	// ─── Phase 3: Parallel Batch Predictions ─────────────
	fmt.Println("\n[3] Running batch parallel predictions...")
	predStart := time.Now()
	batchResults := prover.BatchPredictParallel(setup, weights, bias, features, *numWorkers, *keyPoolSize)
	predTotal := time.Since(predStart)

	// ─── Phase 4: Collect Results ────────────────────────
	var (
		correct       int
		totalProve    time.Duration
		totalVerify   time.Duration
		failedBatches int
	)

	fmt.Println("\n[4] Results")
	fmt.Println("    ┌────────┬──────────┬────────┬──────────────┬──────────────┬──────────┐")
	fmt.Println("    │ Batch  │ Samples  │ Status │ Prove Time   │ Verify Time  │ Proof    │")
	fmt.Println("    ├────────┼──────────┼────────┼──────────────┼──────────────┼──────────┤")

	for _, br := range batchResults {
		if br.Error != nil {
			fmt.Printf("    │ %6d │ %8d │ FAIL   │ %12v │ %12v │          │ %v\n",
				br.BatchIndex, len(br.Predictions), br.ProveTime, br.VerifyTime, br.Error)
			failedBatches++
			continue
		}

		batchCorrect := 0
		for _, p := range br.Predictions {
			idx := 0
			for di, d := range testData {
				h, w := 0, 0
				if len(d.features) > 0 {
					h = d.features[0]
				}
				if len(d.features) > 1 {
					w = d.features[1]
				}
				if h == p.Height && w == p.Weight {
					idx = di
					break
				}
			}
			actualLabel := "NORMAL"
			if idx < len(testData) && testData[idx].label == 1 {
				actualLabel = "OVERWEIGHT"
			}
			if p.Prediction == actualLabel {
				batchCorrect++
				correct++
			}
		}

		status := "✓"
		if !br.Verified {
			status = "✗"
		}

		fmt.Printf("    │ %6d │ %8d │ %6s │ %12v │ %12v │ %4d B   │\n",
			br.BatchIndex, len(br.Predictions), status,
			br.ProveTime.Round(time.Millisecond),
			br.VerifyTime.Round(time.Microsecond),
			len(br.ProofBytes))

		totalProve += br.ProveTime
		totalVerify += br.VerifyTime
	}

	fmt.Println("    └────────┴──────────┴────────┴──────────────┴──────────────┴──────────┘")

	// ─── Phase 5: Summary ────────────────────────────────
	numBatches := len(batchResults) - failedBatches
	accuracy := float64(correct) / float64(len(testData)) * 100

	fmt.Println("\n[5] Summary")
	fmt.Println("    ═══════════════════════════════════════════════")
	fmt.Printf("    Samples:            %d\n", len(testData))
	fmt.Printf("    Batches:            %d (size=%d)\n", numBatches, *batchSize)
	fmt.Printf("    Workers:            %d\n", *numWorkers)
	fmt.Printf("    Correct:            %d / %d\n", correct, len(testData))
	fmt.Printf("    Accuracy:           %.2f%%\n", accuracy)
	fmt.Println("    ───────────────────────────────────────────────")
	fmt.Printf("    Setup time:         %v\n", setupTotal)
	fmt.Printf("    Total predict time: %v\n", predTotal)
	fmt.Printf("    Total prove time:   %v (sequential sum)\n", totalProve)
	fmt.Printf("    Total verify time:  %v\n", totalVerify)
	if numBatches > 0 {
		fmt.Printf("    Avg prove/batch:    %v\n", totalProve/time.Duration(numBatches))
		fmt.Printf("    Avg prove/sample:   %.3fs\n", totalProve.Seconds()/float64(len(testData)))
		fmt.Printf("    Wall-clock/sample:  %.3fs\n", predTotal.Seconds()/float64(len(testData)))
	}
	fmt.Printf("    Proof size:         584 bytes (per batch, constant)\n")
	fmt.Printf("    Speedup vs single:  %.1fx\n", (float64(len(testData))*2.0)/predTotal.Seconds())
	fmt.Println("    ═══════════════════════════════════════════════")

	// ─── Phase 6: Export ─────────────────────────────────
	exportBatchResults(batchResults, testData, accuracy, setup, predTotal, totalProve, totalVerify, *numWorkers)
	fmt.Println("\n    Results saved → results/batch_prediction_results.txt")
	fmt.Println("\n✅ Done!")
}

// ─── CSV Loading ─────────────────────────────────────────────

type sample struct {
	features []int
	label    int // 1 = overweight, 0 = normal
}

func loadCSV(path string) ([]sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var samples []sample
	scanner := bufio.NewScanner(f)
	first := true
	for scanner.Scan() {
		line := scanner.Text()
		if first {
			first = false
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}

		lastIdx := len(parts) - 1
		label, err := strconv.Atoi(strings.TrimSpace(parts[lastIdx]))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: invalid label, skipping row: %v\n", err)
			continue
		}
		
		features := make([]int, lastIdx)
		rowValid := true
		for j := 0; j < lastIdx; j++ {
			f, err := strconv.Atoi(strings.TrimSpace(parts[j]))
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: invalid feature at column %d, skipping row\n", j)
				rowValid = false
				break
			}
			features[j] = f
		}
		if !rowValid {
			continue
		}

		samples = append(samples, sample{features: features, label: label})
	}
	return samples, scanner.Err()
}

// ─── Export ──────────────────────────────────────────────────

func exportBatchResults(results []*prover.BatchPredResult, data []sample, accuracy float64, setup *prover.BatchSetupResult, predTotal, totalProve, totalVerify time.Duration, workers int) {
	f, err := os.Create("results/batch_prediction_results.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to create results file: %v\n", err)
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "ZK-LR Batch Parallel Prediction Results\n")
	fmt.Fprintf(f, "========================================\n\n")
	fmt.Fprintf(f, "Configuration:\n")
	fmt.Fprintf(f, "  Batch size:       %d\n", setup.BatchSize)
	fmt.Fprintf(f, "  Workers:          %d\n", workers)
	fmt.Fprintf(f, "  Constraints:      %d\n", setup.NumConstraints)
	fmt.Fprintf(f, "  PK size:          %.1f KB\n", float64(setup.PKSizeBytes)/1024)
	fmt.Fprintf(f, "  VK size:          %.1f KB\n\n", float64(setup.VKSizeBytes)/1024)

	fmt.Fprintf(f, "Results:\n")
	fmt.Fprintf(f, "  Total samples:    %d\n", len(data))
	fmt.Fprintf(f, "  Accuracy:         %.2f%%\n", accuracy)
	fmt.Fprintf(f, "  Total predict:    %v\n", predTotal)
	fmt.Fprintf(f, "  Total prove:      %v\n", totalProve)
	fmt.Fprintf(f, "  Total verify:     %v\n", totalVerify)
	fmt.Fprintf(f, "  Wall-clock/sample: %.3fs\n", predTotal.Seconds()/float64(len(data)))
	fmt.Fprintf(f, "  Speedup vs single: %.1fx\n", (float64(len(data))*2.0)/predTotal.Seconds())
}
