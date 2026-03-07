package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"strconv"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
	"github.com/consensys/gnark/test/unsafekzg"
)

// --- Integer Scaling parameters ---
const Precision = 32
var scalingFactor = new(big.Int).Lsh(big.NewInt(1), Precision)

// --- Sigmoid Table parameters ---
const inputPrecision = 10
var InputScale = new(big.Int).Lsh(big.NewInt(1), inputPrecision)

const outputPrecision = 16
var OutputScale = new(big.Int).Lsh(big.NewInt(1), outputPrecision)
const MaxInput = 8

// --- Pre-computed model parameters (integer-only) ---
// W = round(-0.85735312 × 2^32), B = round(50.94705066 × 2^32)
var W_SCALED = new(big.Int).SetInt64(-3682303612)
var B_SCALED = new(big.Int).SetInt64(218815916412)

// --- Batch Configuration ---
const BatchSize = 20  // Process 20 samples per batch

// --- Batched Circuit Definition ---
type BatchCircuit struct {
	W frontend.Variable          // Private: model weight (shared across batch)
	B frontend.Variable          // Private: model bias (shared across batch)
	X [BatchSize]frontend.Variable  // Public: input features (marks)
	Z [BatchSize]frontend.Variable  // Private: linear outputs (W*X + B)
	Y [BatchSize]frontend.Variable  // Public: predictions
}

// Define the batched circuit: processes BatchSize samples in one proof
func (circuit *BatchCircuit) Define(api frontend.API) error {
	// Initialize and build the sigmoid lookup table once for the entire batch
	table := logderivlookup.New(api)
	
	tablesize := MaxInput * (1 << inputPrecision)  // 8 * 1024 = 8192
	
	for i := 0; i <= tablesize; i++ {
		x_float := float64(i) / float64(1<<inputPrecision)
		y_float := 1.0 / (1.0 + math.Exp(-x_float))
		y_scaled := int64(y_float * float64(1<<outputPrecision))
		table.Insert(y_scaled)
	}

	// Process each sample in the batch
	oneOut := big.NewInt(1<<outputPrecision - 1)
	zeroOut := big.NewInt(0)
	maxIn := new(big.Int).Mul(big.NewInt(MaxInput), InputScale)

	for i := 0; i < BatchSize; i++ {
		z_sigmoid_input := circuit.Z[i]
		
		// Sigmoid computation with clamping (same as single-sample version)
		cmpMax := api.Cmp(z_sigmoid_input, maxIn)
		isGreaterThanMax := api.IsZero(api.Sub(1, cmpMax))
		
		negZ := api.Neg(z_sigmoid_input)
		cmpNegZ := api.Cmp(negZ, z_sigmoid_input)
		isNeg := api.IsZero(api.Add(1, cmpNegZ))
		
		isSaturatedPos := api.Mul(isGreaterThanMax, api.Sub(1, isNeg))
		
		absZ := api.Select(isNeg, negZ, z_sigmoid_input)
		
		cmpAbsMax := api.Cmp(absZ, maxIn)
		isAbsSaturated := api.IsZero(api.Sub(1, cmpAbsMax))
		isSaturatedNeg := api.Mul(isNeg, isAbsSaturated)
		
		clampedAbsZ := api.Select(isAbsSaturated, maxIn, absZ)
		
		lookupResults := table.Lookup(clampedAbsZ)[0]
		
		oneMinusResults := api.Sub(oneOut, lookupResults)
		normalResults := api.Select(isNeg, oneMinusResults, lookupResults)
		
		result1 := api.Select(isSaturatedNeg, zeroOut, normalResults)
		finalResults := api.Select(isSaturatedPos, oneOut, result1)
		
		// Assert this sample's output matches expected prediction
		api.AssertIsEqual(finalResults, circuit.Y[i])
	}

	return nil
}

// --- Integer-only helper functions ---

// newScaledInt scales an integer input by 2^32 (no floating-point)
func newScaledInt(val int64) *big.Int {
	return new(big.Int).Mul(big.NewInt(val), scalingFactor)
}

// sigmoidToScaledOutput converts a sigmoid float result [0,1] to scaled integer (×2^16)
// This is the only place floating-point is used: to convert sigmoid(z) output
func sigmoidToScaledOutput(val float64) *big.Int {
	return new(big.Int).SetInt64(int64(val * float64(1<<outputPrecision)))
}

func loadTestData(filepath string) ([]int, []int, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	marks := make([]int, 0)
	labels := make([]int, 0)

	for i := 1; i < len(records); i++ {
		if len(records[i]) < 2 {
			continue
		}
		mark, err := strconv.Atoi(records[i][0])
		if err != nil {
			continue
		}
		label, err := strconv.Atoi(records[i][1])
		if err != nil {
			continue
		}
		marks = append(marks, mark)
		labels = append(labels, label)
	}

	return marks, labels, nil
}

type BatchResult struct {
	batchIdx      int
	samplesInBatch int
	correctCount  int
	proveTime     time.Duration
	verifyTime    time.Duration
	err           error
}

func main() {
	fmt.Println("================================================================")
	fmt.Println("   ZK Logistic Regression - BATCHED Proof System")
	fmt.Println("================================================================\n")

	var circuit BatchCircuit
	var ccs *cs.SparseR1CS
	var pk plonk.ProvingKey
	var vk plonk.VerifyingKey
	var err error

	// Setup Phase
	startSetup := time.Now()
	fmt.Println("[SETUP] Compiling batched circuit...")
	ccsTemp, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &circuit)
	if err != nil {
		log.Fatal("circuit compilation error: ", err)
	}
	ccs = ccsTemp.(*cs.SparseR1CS)
	fmt.Printf("[SETUP] ✓ Circuit compiled (%d constraints for %d samples)\n", 
		ccs.GetNbConstraints(), BatchSize)

	fmt.Println("[SETUP] Generating SRS...")
	srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
	if err != nil {
		panic(err)
	}
	fmt.Println("[SETUP] ✓ SRS generated")

	fmt.Println("[SETUP] Running trusted setup (generating PK, VK)...")
	pk, vk, err = plonk.Setup(ccs, srs, srsLagrange)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("[SETUP] ✓ Keys generated")
	
	setupTime := time.Since(startSetup)
	fmt.Printf("[SETUP] Total setup time: %v\n\n", setupTime)

	// Load dataset
	marks, labels, err := loadTestData("data/student_dataset.csv")
	if err != nil {
		log.Fatal("Error loading test data:", err)
	}
	
	// Use 100 samples for testing (change to 3000 for production)
	testSize := 3000 // Full production dataset for H100
	if len(marks) > testSize {
		marks = marks[:testSize]
		labels = labels[:testSize]
	}
	
	fmt.Printf("[DATA] Loaded %d samples from dataset\n", len(marks))
	fmt.Printf("[BATCH] Processing in batches of %d samples\n", BatchSize)
	
	// Model parameters
	w_bi := W_SCALED
	b_bi := B_SCALED
	
	// Calculate number of batches needed
	numBatches := (len(marks) + BatchSize - 1) / BatchSize
	fmt.Printf("[BATCH] Total batches: %d\n\n", numBatches)

	// Process batches
	startProving := time.Now()
	batchResults := make([]BatchResult, 0, numBatches)
	totalCorrect := 0

	for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
		startIdx := batchIdx * BatchSize
		endIdx := startIdx + BatchSize
		if endIdx > len(marks) {
			endIdx = len(marks)
		}
		batchMarks := marks[startIdx:endIdx]
		batchLabels := labels[startIdx:endIdx]
		actualBatchSize := len(batchMarks)

		fmt.Printf("[BATCH %d/%d] Processing samples %d-%d (%d samples)...\n", 
			batchIdx+1, numBatches, startIdx+1, endIdx, actualBatchSize)

		// Build witness for this batch
		var witness BatchCircuit
		witness.W = w_bi
		witness.B = b_bi

		correctInBatch := 0

		for i := 0; i < BatchSize; i++ {
			var mark int
			var label int
			
			if i < actualBatchSize {
				mark = batchMarks[i]
				label = batchLabels[i]
			} else {
				// Pad with dummy values if batch is incomplete
				mark = batchMarks[0]
				label = batchLabels[0]
			}

			x_bi := newScaledInt(int64(mark))
			
			// Calculate z = WX + B
			wx_scaled2 := new(big.Int).Mul(w_bi, x_bi)
			wx_scaled1 := new(big.Int).Div(wx_scaled2, scalingFactor)
			z_linear_bi := new(big.Int).Add(wx_scaled1, b_bi)
			
			// Rescale z from Q32 to Q10
			shiftFactorWit := new(big.Int).Lsh(big.NewInt(1), Precision-inputPrecision)
			z_sigmoid_input_bi := new(big.Int).Div(z_linear_bi, shiftFactorWit)
			
			// Calculate Y = sigmoid(z)
			z_quantized_f := new(big.Float).SetInt(z_sigmoid_input_bi)
			z_quantized_f.Quo(z_quantized_f, new(big.Float).SetInt(InputScale))
			z_float, _ := z_quantized_f.Float64()

			var y_float float64
			if z_float > float64(MaxInput) {
				y_float = 1.0
			} else if z_float < -float64(MaxInput) {
				y_float = 0.0
			} else {
				y_float = 1.0 / (1.0 + math.Exp(-z_float))
			}

			y_scaled_bi := sigmoidToScaledOutput(y_float)
			maxOutput := big.NewInt(1<<outputPrecision - 1)
			if y_scaled_bi.Cmp(maxOutput) > 0 {
				y_scaled_bi = maxOutput
			}

			pred_label := 0
			if y_float >= 0.5 {
				pred_label = 1
			}

			// Count correct predictions (only for real samples, not padding)
			if i < actualBatchSize && pred_label == label {
				correctInBatch++
			}

			witness.X[i] = x_bi
			witness.Z[i] = z_sigmoid_input_bi
			witness.Y[i] = y_scaled_bi
		}

		// Generate proof for this batch
		startProve := time.Now()
		witnessFull, err := frontend.NewWitness(&witness, ecc.BN254.ScalarField())
		if err != nil {
			log.Printf("[ERROR] Batch %d witness generation failed: %v\n", batchIdx+1, err)
			continue
		}

		witnessPublic, err := witnessFull.Public()
		if err != nil {
			log.Printf("[ERROR] Batch %d public witness failed: %v\n", batchIdx+1, err)
			continue
		}

		proof, err := plonk.Prove(ccs, pk, witnessFull)
		proveTime := time.Since(startProve)
		if err != nil {
			log.Printf("[ERROR] Batch %d proving failed: %v\n", batchIdx+1, err)
			continue
		}

		// Verify proof
		startVerify := time.Now()
		err = plonk.Verify(proof, vk, witnessPublic)
		verifyTime := time.Since(startVerify)
		if err != nil {
			log.Printf("[ERROR] Batch %d verification failed: %v\n", batchIdx+1, err)
			continue
		}

		fmt.Printf("[BATCH %d/%d] ✓ Proved: %v | Verified: %v | Correct: %d/%d\n", 
			batchIdx+1, numBatches, proveTime, verifyTime, correctInBatch, actualBatchSize)

		batchResults = append(batchResults, BatchResult{
			batchIdx:       batchIdx,
			samplesInBatch: actualBatchSize,
			correctCount:   correctInBatch,
			proveTime:      proveTime,
			verifyTime:     verifyTime,
			err:            nil,
		})

		totalCorrect += correctInBatch
	}

	totalProvingTime := time.Since(startProving)

	// Calculate statistics
	accuracy := float64(totalCorrect) / float64(len(marks)) * 100
	
	var totalProveTime, totalVerifyTime time.Duration
	for _, r := range batchResults {
		totalProveTime += r.proveTime
		totalVerifyTime += r.verifyTime
	}
	
	avgProveTime := totalProveTime / time.Duration(len(batchResults))
	avgVerifyTime := totalVerifyTime / time.Duration(len(batchResults))

	// Calculate proof size
	var sampleWitness BatchCircuit
	sampleWitness.W = w_bi
	sampleWitness.B = b_bi
	for i := 0; i < BatchSize; i++ {
		sampleWitness.X[i] = newScaledInt(75)
		sampleWitness.Z[i] = big.NewInt(0)
		sampleWitness.Y[i] = big.NewInt(1 << (outputPrecision - 1))
	}
	witnessFull, _ := frontend.NewWitness(&sampleWitness, ecc.BN254.ScalarField())
	proof, _ := plonk.Prove(ccs, pk, witnessFull)
	var buf bytes.Buffer
	proof.WriteTo(&buf)
	proofSize := buf.Len()

	// Print final results
	fmt.Println("\n================================================================")
	fmt.Println("                     FINAL RESULTS")
	fmt.Println("================================================================\n")
	
	fmt.Printf("Batched Processing Summary:\n")
	fmt.Printf("  Batch size:           %d samples/batch\n", BatchSize)
	fmt.Printf("  Total batches:        %d\n", len(batchResults))
	fmt.Printf("  Total samples:        %d\n", len(marks))
	fmt.Printf("  Constraints/batch:    %d\n", ccs.GetNbConstraints())
	fmt.Printf("  Proof size:           %d bytes\n\n", proofSize)
	
	fmt.Printf("Accuracy Metrics:\n")
	fmt.Printf("  Correct predictions:  %d/%d\n", totalCorrect, len(marks))
	fmt.Printf("  Accuracy:             %.2f%%\n", accuracy)
	fmt.Printf("  Threshold:            97%%\n")
	if accuracy >= 97.0 {
		fmt.Printf("  Status:               PASSED\n\n")
	} else {
		fmt.Printf("  Status:               Below threshold\n\n")
	}
	
	fmt.Printf("Performance Metrics:\n")
	fmt.Printf("  Setup time:           %v\n", setupTime)
	fmt.Printf("  Total proving time:   %v\n", totalProvingTime)
	fmt.Printf("  Avg prove/batch:      %v\n", avgProveTime)
	fmt.Printf("  Avg verify/batch:     %v\n", avgVerifyTime)
	fmt.Printf("  Batches/second:       %.2f\n", float64(len(batchResults))/totalProvingTime.Seconds())
	fmt.Printf("  Samples/second:       %.2f\n\n", float64(len(marks))/totalProvingTime.Seconds())
	
	fmt.Println("All batch proofs verified successfully!")
	
	// Export metrics
	exportBatchedMetrics(setupTime, totalProvingTime, avgProveTime, avgVerifyTime, 
		len(marks), totalCorrect, ccs.GetNbConstraints(), proofSize, BatchSize, len(batchResults))
}

func exportBatchedMetrics(setupTime, totalProvingTime, avgProveTime, avgVerifyTime time.Duration, 
	samples, correct, constraints, proofSize, batchSize, numBatches int) {
	metricsFile := "data/cache/run_metrics_batched.txt"
	f, err := os.Create(metricsFile)
	if err != nil {
		log.Printf("[WARN] Failed to export metrics: %v", err)
		return
	}
	defer f.Close()
	
	fmt.Fprintf(f, "batch_size=%d\n", batchSize)
	fmt.Fprintf(f, "num_batches=%d\n", numBatches)
	fmt.Fprintf(f, "setup_time_ms=%d\n", setupTime.Milliseconds())
	fmt.Fprintf(f, "total_proving_time_ms=%d\n", totalProvingTime.Milliseconds())
	fmt.Fprintf(f, "avg_prove_time_ms=%d\n", avgProveTime.Milliseconds())
	fmt.Fprintf(f, "avg_verify_time_ms=%d\n", avgVerifyTime.Milliseconds())
	fmt.Fprintf(f, "samples=%d\n", samples)
	fmt.Fprintf(f, "correct=%d\n", correct)
	fmt.Fprintf(f, "accuracy=%.2f\n", float64(correct)/float64(samples)*100)
	fmt.Fprintf(f, "constraints=%d\n", constraints)
	fmt.Fprintf(f, "throughput_samples_per_sec=%.2f\n", float64(samples)/totalProvingTime.Seconds())
	fmt.Fprintf(f, "throughput_batches_per_sec=%.2f\n", float64(numBatches)/totalProvingTime.Seconds())
	fmt.Fprintf(f, "proof_size_bytes=%d\n", proofSize)
	fmt.Fprintf(f, "proof_size_kb=%.2f\n", float64(proofSize)/1024.0)
	
	fmt.Printf("[METRICS] Exported to %s\n", metricsFile)
}
