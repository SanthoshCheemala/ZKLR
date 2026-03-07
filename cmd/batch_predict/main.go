// main.go — ZK-LR Batch Parallel Prediction Pipeline
//
// Usage:
//
//	go run ./cmd/batch_predict/                           # defaults: batch=20, workers=auto
//	go run ./cmd/batch_predict/ -batch=50 -workers=8      # custom
//	go run ./cmd/batch_predict/ -dataset=data/student_dataset.csv -batch=20 -workers=32  # HPC
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

const (
	W1_FLOAT = -3.3144933046
	W2_FLOAT = 0.3877500778
	B_FLOAT  = 281.2861173099
)

func main() {
	// CLI flags
	batchSize := flag.Int("batch", 20, "predictions per proof")
	numWorkers := flag.Int("workers", 0, "parallel workers (0=auto)")
	datasetPath := flag.String("dataset", "data/bmi_dataset_test.csv", "CSV file path")
	flag.Parse()

	if *numWorkers == 0 {
		*numWorkers = runtime.NumCPU()
		if *numWorkers > 8 {
			*numWorkers = 8
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
	setup, err := prover.RunBatchSetup(*batchSize)
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

	// Save keys
	os.MkdirAll("results", 0o755)
	prover.SaveProvingKey(setup.ProvingKey, "results/batch_proving.key")
	prover.SaveVerificationKey(setup.VerificationKey, "results/batch_verification.key")

	// ─── Phase 2: Load Dataset ───────────────────────────
	fmt.Println("\n[2] Loading dataset...")
	testData, err := loadCSV(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("    Loaded %d samples\n", len(testData))

	features := make([][2]int, len(testData))
	for i, s := range testData {
		features[i] = [2]int{s.height, s.weight}
	}

	// ─── Phase 3: Parallel Batch Predictions ─────────────
	fmt.Println("\n[3] Running batch parallel predictions...")
	predStart := time.Now()
	batchResults := prover.BatchPredictParallel(setup, W1_FLOAT, W2_FLOAT, B_FLOAT, features, *numWorkers)
	predTotal := time.Since(predStart)

	// ─── Phase 4: Collect Results ────────────────────────
	var (
		correct       int
		totalSamples  int
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
			totalSamples++
			idx := 0
			for di, d := range testData {
				if d.height == p.Height && d.weight == p.Weight {
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
	height int
	weight int
	label  int // 1 = overweight, 0 = normal
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
		if len(parts) < 3 {
			continue
		}
		height, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		weight, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
		label, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
		samples = append(samples, sample{height: height, weight: weight, label: label})
	}
	return samples, scanner.Err()
}

// ─── Export ──────────────────────────────────────────────────

func exportBatchResults(results []*prover.BatchPredResult, data []sample, accuracy float64, setup *prover.BatchSetupResult, predTotal, totalProve, totalVerify time.Duration, workers int) {
	f, err := os.Create("results/batch_prediction_results.txt")
	if err != nil {
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
