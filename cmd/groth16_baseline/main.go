// cmd/groth16_baseline — Groth16 comparison point for the head-to-head table (T6).
//
// Compiles the SAME BatchLabelCircuit to R1CS, runs Groth16 setup/prove/verify
// on the same witness, and reports constraints, timings and proof size.
// Together with the PLONK pipeline this gives the universal-vs-per-circuit
// setup comparison on identical statements.
//
// NOTE: groth16.Setup here is a dev setup (toxic waste in-process), matching
// how unsafekzg is used on the PLONK side — fine for timing, not for deployment.
//
// Usage:
//
//	go run ./cmd/groth16_baseline -batch=20 -dataset=data/wbc_test.csv \
//	   -weights="..." -bias=-11.07 -runs=3
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"

	"github.com/santhoshcheemala/ZKLR/circuit"
	"github.com/santhoshcheemala/ZKLR/prover"
)

func main() {
	batchSize := flag.Int("batch", 20, "predictions per proof")
	datasetPath := flag.String("dataset", "", "CSV file (header row; last column = label)")
	weightsFlag := flag.String("weights", "", "comma-separated model weights (required)")
	biasFlag := flag.Float64("bias", 0.0, "model bias")
	runs := flag.Int("runs", 3, "prove/verify repetitions (median-friendly)")
	flag.Parse()

	if *weightsFlag == "" || *datasetPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -weights and -dataset are required")
		os.Exit(1)
	}
	weights, err := parseWeights(*weightsFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	features, err := loadFeatures(*datasetPath, len(weights))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(features) > *batchSize {
		features = features[:*batchSize]
	}

	fmt.Println("============================================================")
	fmt.Println("  Groth16 Baseline — same BatchLabelCircuit, R1CS backend")
	fmt.Println("============================================================")
	fmt.Printf("  batch=%d features=%d runs=%d\n", *batchSize, len(weights), *runs)

	// ─── Compile to R1CS ───
	start := time.Now()
	c := circuit.NewBatchLabelCircuit(*batchSize, len(weights))
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c)
	if err != nil {
		fmt.Fprintf(os.Stderr, "R1CS compile failed: %v\n", err)
		os.Exit(1)
	}
	compileTime := time.Since(start)
	fmt.Printf("  R1CS constraints: %d (compile %v)\n", ccs.GetNbConstraints(), compileTime)

	// ─── Setup ───
	start = time.Now()
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Groth16 setup failed: %v\n", err)
		os.Exit(1)
	}
	setupTime := time.Since(start)
	fmt.Printf("  Setup: %v\n", setupTime)

	// ─── Witness ───
	assignment := prover.ComputeBatchLabelWitness(weights, *biasFlag, features, *batchSize)
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		fmt.Fprintf(os.Stderr, "witness failed: %v\n", err)
		os.Exit(1)
	}
	publicWitness, _ := witness.Public()

	// ─── Prove / Verify (repeated) ───
	var proofSize int
	proveTimes := make([]time.Duration, 0, *runs)
	verifyTimes := make([]time.Duration, 0, *runs)
	for i := 0; i < *runs; i++ {
		start = time.Now()
		proof, err := groth16.Prove(ccs, pk, witness)
		proveTimes = append(proveTimes, time.Since(start))
		if err != nil {
			fmt.Fprintf(os.Stderr, "prove failed: %v\n", err)
			os.Exit(1)
		}

		var buf bytes.Buffer
		proof.WriteTo(&buf)
		proofSize = buf.Len()

		start = time.Now()
		err = groth16.Verify(proof, vk, publicWitness)
		verifyTimes = append(verifyTimes, time.Since(start))
		if err != nil {
			fmt.Fprintf(os.Stderr, "verify failed: %v\n", err)
			os.Exit(1)
		}
	}

	var vkBuf bytes.Buffer
	vk.WriteTo(&vkBuf)

	fmt.Println("------------------------------------------------------------")
	fmt.Printf("  Prove times:   %v\n", proveTimes)
	fmt.Printf("  Verify times:  %v\n", verifyTimes)
	fmt.Printf("  Proof size:    %d bytes\n", proofSize)
	fmt.Printf("  VK size:       %d bytes\n", vkBuf.Len())
	fmt.Printf("  Prove/sample:  %.3fs (last run / batch)\n",
		proveTimes[len(proveTimes)-1].Seconds()/float64(*batchSize))
	fmt.Println("------------------------------------------------------------")
	fmt.Println("  Reminder: Groth16 setup is per-circuit (re-run for every batch")
	fmt.Println("  size / feature count); PLONK reuses one universal SRS.")
}

func parseWeights(s string) ([]float64, error) {
	parts := strings.Split(s, ",")
	out := make([]float64, len(parts))
	for i, p := range parts {
		w, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight %q: %v", p, err)
		}
		out[i] = w
	}
	return out, nil
}

func loadFeatures(path string, numFeatures int) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var rows [][]int
	scanner := bufio.NewScanner(f)
	first := true
	for scanner.Scan() {
		if first {
			first = false
			continue
		}
		parts := strings.Split(scanner.Text(), ",")
		if len(parts) < numFeatures+1 {
			continue
		}
		row := make([]int, numFeatures)
		ok := true
		for j := 0; j < numFeatures; j++ {
			v, err := strconv.Atoi(strings.TrimSpace(parts[j]))
			if err != nil {
				ok = false
				break
			}
			row[j] = v
		}
		if ok {
			rows = append(rows, row)
		}
	}
	return rows, scanner.Err()
}
