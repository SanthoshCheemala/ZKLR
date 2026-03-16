// main.go — ZK Logistic Regression Prediction Pipeline
//
// This is the main entry point that ties together:
//  1. Setup:   Compile circuit → Generate SRS → Create keys
//  2. Predict:  For each student, compute witness → prove → verify
//  3. Report:  Export results and metrics
//
// Usage: go run main.go
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/santhoshcheemala/ZKLR/prover"
)


func main() {
	fmt.Println("============================================================")
	fmt.Println("  ZK Logistic Regression — Prediction Pipeline")
	fmt.Println("============================================================")

	datasetPath := flag.String("dataset", "data/bmi_dataset_test.csv", "CSV file path")
	weightsFlag := flag.String("weights", "-3.3144933046,0.3877500778", "comma-separated model weights")
	biasFlag := flag.Float64("bias", 281.2861173099, "model bias")
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

	// ─── Phase 1: Setup ──────────────────────────────────
	fmt.Println("\n[1] Running ZK Setup (compile → SRS → keys)...")
	setupStart := time.Now()
	setup, err := prover.RunSetup(len(weights))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Setup failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("    Constraints:   %d\n", setup.NumConstraints)
	fmt.Printf("    Compile time:  %v\n", setup.CompileTime)
	fmt.Printf("    Setup time:    %v\n", setup.SetupTime)
	fmt.Printf("    PK size:       %.1f KB\n", float64(setup.PKSizeBytes)/1024)
	fmt.Printf("    VK size:       %.1f KB\n", float64(setup.VKSizeBytes)/1024)
	fmt.Printf("    Total setup:   %v\n", time.Since(setupStart))

	// Save keys
	os.MkdirAll("results", 0o755)
	prover.SaveProvingKey(setup.ProvingKey, "results/proving.key")
	prover.SaveVerificationKey(setup.VerificationKey, "results/verification.key")
	fmt.Println("    Keys saved → results/proving.key, results/verification.key")

	// ─── Phase 2: Load Test Dataset ──────────────────────
	fmt.Println("\n[2] Loading test dataset...")
	testData, err := loadCSV(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load test data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("    Loaded %d test samples\n", len(testData))

	// ─── Phase 3: ZK Predictions ─────────────────────────
	fmt.Println("\n[3] Running ZK predictions...")
	fmt.Println("    ┌─────────┬────────────┬────────────┬────────────┬──────────┬──────────┐")
	fmt.Println("    │ Hgt,Wgt │ Prob       │ Predicted  │ Actual     │ Prove    │ Verify   │")
	fmt.Println("    ├─────────┼────────────┼────────────┼────────────┼──────────┼──────────┤")

	var (
		correct      int
		totalProve   time.Duration
		totalVerify  time.Duration
		results      []*prover.PredictionResult
	)

	for i, sample := range testData {
		result, err := prover.Predict(setup, weights, bias, sample.features)
		if err != nil {
			fmt.Fprintf(os.Stderr, "    Sample %d failed: %v\n", i, err)
			continue
		}

		results = append(results, result)
		totalProve += result.ProveTime
		totalVerify += result.VerifyTime

		// Check correctness against ground truth
		actualLabel := "NORMAL"
		if sample.label == 1 {
			actualLabel = "OVERWEIGHT"
		}
		match := result.Prediction == actualLabel
		if match {
			correct++
		}

		matchStr := "✓"
		if !match {
			matchStr = "✗"
		}

		// Display first two features if available
		f1, f2 := 0, 0
		if len(sample.features) > 0 { f1 = sample.features[0] }
		if len(sample.features) > 1 { f2 = sample.features[1] }

		fmt.Printf("    │ %3d,%3d │ %8.4f   │ %-10s │ %-10s │ %6.2fs  │ %6.2fms │ %s\n",
			f1, f2,
			result.Probability,
			result.Prediction,
			actualLabel,
			result.ProveTime.Seconds(),
			float64(result.VerifyTime.Microseconds())/1000,
			matchStr,
		)
	}

	fmt.Println("    └───────┴────────────┴────────────┴────────────┴──────────┴──────────┘")

	// ─── Phase 4: Summary ────────────────────────────────
	accuracy := float64(correct) / float64(len(testData)) * 100

	fmt.Println("\n[4] Summary")
	fmt.Println("    ═══════════════════════════════════════")
	fmt.Printf("    Samples:         %d\n", len(testData))
	fmt.Printf("    Correct:         %d / %d\n", correct, len(testData))
	fmt.Printf("    Accuracy:        %.2f%%\n", accuracy)
	fmt.Printf("    Total prove:     %v\n", totalProve)
	fmt.Printf("    Total verify:    %v\n", totalVerify)
	fmt.Printf("    Avg prove/sample: %.2fs\n", totalProve.Seconds()/float64(len(results)))
	fmt.Printf("    Avg verify/sample: %.3fms\n", float64(totalVerify.Microseconds())/float64(len(results))/1000)
	fmt.Printf("    Proof size:      584 bytes (constant)\n")
	fmt.Println("    ═══════════════════════════════════════")

	// ─── Phase 5: Export Results ─────────────────────────
	exportResults(results, accuracy, setup, totalProve, totalVerify)
	fmt.Println("\n    Results saved → results/prediction_results.txt")
	fmt.Println("\n✅ Done!")
}

// ─── CSV Loading ─────────────────────────────────────────────

type sample struct {
	features []int
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
			continue // skip header
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		
		lastIdx := len(parts) - 1
		label, _ := strconv.Atoi(strings.TrimSpace(parts[lastIdx]))
		
		features := make([]int, lastIdx)
		for j := 0; j < lastIdx; j++ {
			features[j], _ = strconv.Atoi(strings.TrimSpace(parts[j]))
		}
		
		samples = append(samples, sample{features: features, label: label})
	}
	return samples, scanner.Err()
}

// ─── Results Export ──────────────────────────────────────────

func exportResults(results []*prover.PredictionResult, accuracy float64, setup *prover.SetupResult, totalProve, totalVerify time.Duration) {
	f, err := os.Create("results/prediction_results.txt")
	if err != nil {
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "ZK Logistic Regression — Prediction Results\n")
	fmt.Fprintf(f, "============================================\n\n")
	fmt.Fprintf(f, "Circuit:\n")
	fmt.Fprintf(f, "  Constraints:    %d\n", setup.NumConstraints)
	fmt.Fprintf(f, "  Compile time:   %v\n", setup.CompileTime)
	fmt.Fprintf(f, "  Setup time:     %v\n", setup.SetupTime)
	fmt.Fprintf(f, "  PK size:        %.1f KB\n", float64(setup.PKSizeBytes)/1024)
	fmt.Fprintf(f, "  VK size:        %.1f KB\n\n", float64(setup.VKSizeBytes)/1024)

	fmt.Fprintf(f, "Predictions:\n")
	fmt.Fprintf(f, "  Total samples:  %d\n", len(results))
	fmt.Fprintf(f, "  Accuracy:       %.2f%%\n", accuracy)
	fmt.Fprintf(f, "  Total prove:    %v\n", totalProve)
	fmt.Fprintf(f, "  Total verify:   %v\n", totalVerify)
	if len(results) > 0 {
		fmt.Fprintf(f, "  Avg prove:      %.3fs\n", totalProve.Seconds()/float64(len(results)))
		fmt.Fprintf(f, "  Avg verify:     %.3fms\n", float64(totalVerify.Microseconds())/float64(len(results))/1000)
	}
	fmt.Fprintf(f, "  Proof size:     584 bytes (constant)\n\n")

	fmt.Fprintf(f, "Per-Sample Results:\n")
	fmt.Fprintf(f, "  Hgt  Wgt  Probability  Prediction  Proved  Verified\n")
	for _, r := range results {
		fmt.Fprintf(f, "  %3d  %3d  %8.4f     %-10s  %.2fs   %.3fms\n",
			r.Height, r.Weight, r.Probability, r.Prediction,
			r.ProveTime.Seconds(), float64(r.VerifyTime.Microseconds())/1000)
	}
}
