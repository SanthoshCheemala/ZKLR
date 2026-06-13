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

// autoBatchSize picks the optimal batch size based on machine specs.
// HPC result: batch=80 was the sweet spot on a 32-core server.
// Scaled down proportionally for laptops/desktops.
func autoBatchSize() int {
	cores := runtime.NumCPU()
	switch {
	case cores >= 16:
		return 80 // HPC / server — confirmed sweet spot
	case cores >= 8:
		return 40 // mid-range workstation (8-15 cores)
	case cores >= 4:
		return 20 // typical laptop (4-7 cores)
	default:
		return 10 // low-end / 2-core machines
	}
}

func main() {
	// CLI flags — batch=0 means auto-detect from machine specs
	batchSize := flag.Int("batch", 0, "predictions per proof (0=auto based on CPU cores)")
	numWorkers := flag.Int("workers", 0, "parallel workers (0=auto: 50% of CPU cores)")
	keyPoolSize := flag.Int("keys", 0, "key pool size for memory control (0=auto, max 16)")
	datasetPath := flag.String("dataset", "data/test_200.csv", "CSV file path")
	weightsFlag := flag.String("weights", "", "comma-separated model weights (required)")
	biasFlag := flag.Float64("bias", 0.0, "model bias")
	modeFlag := flag.String("mode", "label", "public output per sample: 'label' (class bit only, default) or 'prob' (full probability — enables model extraction, use only with trusted verifiers)")
	exportDir := flag.String("export-proofs", "", "directory to export vk/proofs/public witnesses for independent verification (empty = no export)")
	srsPath := flag.String("srs", "", "gnark-serialized BN254 ceremony SRS file (empty = DEV-ONLY locally generated SRS; see docs/TRUSTED_SETUP.md)")
	csvPath := flag.String("csv", "", "append one machine-readable benchmark row to this CSV (header written if new) — for HPC sweeps")
	runID := flag.Int("run", 0, "repetition index recorded in the -csv row (for N-run median/std)")
	flag.Parse()

	mode, err := prover.ParseOutputMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if *weightsFlag == "" {
		fmt.Fprintf(os.Stderr, "Error: -weights flag is required (comma-separated floats, one per feature)\n")
		os.Exit(1)
	}

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

	if *batchSize == 0 {
		*batchSize = autoBatchSize()
		fmt.Printf("  [auto] Detected %d CPU cores → batch=%d\n", runtime.NumCPU(), *batchSize)
	}

	if *numWorkers == 0 {
		// Use 50% of available cores (HPC-tuned)
		*numWorkers = runtime.NumCPU() / 2
		if *numWorkers < 1 {
			*numWorkers = 1
		}
	}

	commitment := prover.ModelCommitment(weights, bias)

	fmt.Println("============================================================")
	fmt.Println("  ZK-LR — Batch Parallel Prediction Pipeline")
	fmt.Println("============================================================")
	fmt.Printf("  Dataset:    %s\n", *datasetPath)
	fmt.Printf("  Batch size: %d\n", *batchSize)
	fmt.Printf("  Workers:    %d (CPUs: %d)\n", *numWorkers, runtime.NumCPU())
	fmt.Printf("  Mode:       %s\n", mode)
	fmt.Printf("  Model commitment: 0x%s\n", commitment.Text(16))
	if mode == prover.ModeProb {
		fmt.Println("  WARNING: prob mode publishes exact probabilities; observers can reconstruct the model from ~d+1 predictions")
	}

	// ─── Phase 1: Setup ──────────────────────────────────
	fmt.Println("\n[1] Batch Setup (compile → SRS → keys)...")
	setupStart := time.Now()
	setup, err := prover.RunBatchSetupFull(*batchSize, len(weights), mode, "results", *srsPath)
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

		// Batches preserve dataset order: prediction i of batch k is sample k*batchSize+i.
		for i, p := range br.Predictions {
			idx := br.BatchIndex*(*batchSize) + i
			if idx >= len(testData) {
				continue
			}
			actualLabel := "NORMAL"
			if testData[idx].label == 1 {
				actualLabel = "OVERWEIGHT"
			}
			if p.Prediction == actualLabel {
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
	if proofSize := measuredProofSize(batchResults); proofSize > 0 {
		fmt.Printf("    Proof size:         %d bytes (per batch, constant)\n", proofSize)
	}
	fmt.Println("    ═══════════════════════════════════════════════")

	// ─── Phase 6: Export ─────────────────────────────────
	exportBatchResults(batchResults, testData, accuracy, setup, predTotal, totalProve, totalVerify, *numWorkers)
	fmt.Println("\n    Results saved → results/batch_prediction_results.txt")

	if *exportDir != "" {
		if err := prover.ExportProofArtifacts(*exportDir, setup, batchResults, commitment); err != nil {
			fmt.Fprintf(os.Stderr, "Proof export failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("    Verify independently: go run ./cmd/verify -dir=%s\n", *exportDir)
	}

	if *csvPath != "" {
		provesPerSample := 0.0
		wallPerSample := 0.0
		if len(testData) > 0 {
			provesPerSample = totalProve.Seconds() / float64(len(testData))
			wallPerSample = predTotal.Seconds() / float64(len(testData))
		}
		appendCSVRow(*csvPath, csvRow{
			Mode: string(mode), Batch: *batchSize, Workers: *numWorkers,
			Keys: *keyPoolSize, Features: len(weights), Samples: len(testData),
			Run: *runID, Constraints: setup.NumConstraints,
			SetupSeconds: setupTotal.Seconds(), PredictSeconds: predTotal.Seconds(),
			ProveTotalSeconds: totalProve.Seconds(),
			ProvePerSample:    provesPerSample, WallPerSample: wallPerSample,
			ProofBytes: measuredProofSize(batchResults), Accuracy: accuracy,
		})
		fmt.Printf("    Benchmark row appended → %s\n", *csvPath)
	}
}

// ─── Machine-readable benchmark CSV (for HPC sweeps) ─────────

type csvRow struct {
	Mode                                            string
	Batch, Workers, Keys, Features, Samples, Run    int
	Constraints                                     int
	SetupSeconds, PredictSeconds, ProveTotalSeconds float64
	ProvePerSample, WallPerSample                   float64
	ProofBytes                                      int
	Accuracy                                        float64
}

const csvHeader = "mode,batch,workers,keys,features,samples,run,constraints," +
	"setup_seconds,predict_seconds,prove_total_seconds,prove_per_sample,wall_per_sample,proof_bytes,accuracy\n"

func appendCSVRow(path string, r csvRow) {
	needHeader := false
	if info, err := os.Stat(path); err != nil || info.Size() == 0 {
		needHeader = true
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: cannot write CSV %q: %v\n", path, err)
		return
	}
	defer f.Close()
	if needHeader {
		f.WriteString(csvHeader)
	}
	fmt.Fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.5f,%.5f,%d,%.2f\n",
		r.Mode, r.Batch, r.Workers, r.Keys, r.Features, r.Samples, r.Run, r.Constraints,
		r.SetupSeconds, r.PredictSeconds, r.ProveTotalSeconds,
		r.ProvePerSample, r.WallPerSample, r.ProofBytes, r.Accuracy)
}

// measuredProofSize returns the proof size from the first successful batch.
func measuredProofSize(results []*prover.BatchPredResult) int {
	for _, br := range results {
		if br.Error == nil && len(br.ProofBytes) > 0 {
			return len(br.ProofBytes)
		}
	}
	return 0
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
	if proofSize := measuredProofSize(results); proofSize > 0 {
		fmt.Fprintf(f, "  Proof size:       %d bytes per batch\n", proofSize)
	}
}
