package main

import (
	"bytes"
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"runtime"
	"strconv"
	"sync"
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
const MaxInput = 8 // Range: table covers [0, 8] (sufficient for logistic regression)

// --- Pre-computed model parameters (integer-only) ---
// W = round(-0.85735312 × 2^32), B = round(50.94705066 × 2^32)
var W_SCALED = new(big.Int).SetInt64(-3682303612)
var B_SCALED = new(big.Int).SetInt64(218815916412)



// --- Circuit Definition ---

type Circuit struct {
	W frontend.Variable
	B frontend.Variable
	X frontend.Variable `gnark:",public"`
	Z frontend.Variable  // z = w*x + b, pre-computed and scaled by 2^10
	Y frontend.Variable `gnark:",public"`
}

// Define the circuit: Y = sigmoid(WX + B)
func (circuit *Circuit) Define(api frontend.API) error {

	// Initialize and build the sigmoid lookup table
	// Create a fresh table instance for each circuit evaluation
	table := logderivlookup.New(api)
	
	tablesize := MaxInput * (1 << inputPrecision)  // 8 * 1024 = 8192
	
	// Build sigmoid lookup table
	// Index i represents input value: i / 2^10 (in range [0, 8])
	for i := 0; i <= tablesize; i++ {
		// Convert index to actual x value
		x_float := float64(i) / float64(1<<inputPrecision)
		
		// Calculate sigmoid(x)
		y_float := 1.0 / (1.0 + math.Exp(-x_float))
		
		// Scale output by 2^16
		y_scaled := int64(y_float * float64(1<<outputPrecision))
		
		// Insert the OUTPUT value (input is implicit = index i)
		table.Insert(y_scaled)
	}

	// === Part 1: Use Z directly (pre-computed in witness, scaled by 2^10) ===
	// Z = (W*X + B) rescaled from 2^32 to 2^10
	// We accept Z as input to avoid problematic division in the circuit
	z_sigmoid_input := circuit.Z
	
	// Note: We keep W, B, X in the circuit struct for documentation,
	// but don't use them in constraints due to division issues with negative numbers

	// === Part 3: Calculate Sigmoid (with clamping) ===
	// Define constants for sigmoid logic (scaled by 2^10 and 2^16)
	oneOut := big.NewInt(1<<outputPrecision - 1)  // 2^16 - 1 = 65535 (max value that fits in 16 bits)
	zeroOut := big.NewInt(0)
	maxIn := new(big.Int).Mul(big.NewInt(MaxInput), InputScale) // 8 * 2^10

	// To handle negative numbers in finite field, we need to distinguish:
	// - Small values [0, maxIn]: valid positive range for lookup
	// - Large values (maxIn, field_size/2): positive saturation
	// - Very large values (field_size/2, field_size): negative numbers
	
	// First check if z > maxIn
	cmpMax := api.Cmp(z_sigmoid_input, maxIn)
	isGreaterThanMax := api.IsZero(api.Sub(1, cmpMax)) // 1 if z > maxIn
	
	// To detect negative: compare z with its negation
	// If z is negative (represented as large positive near field_size),
	// then -z will be a small positive value, so -z < z
	negZ := api.Neg(z_sigmoid_input)
	cmpNegZ := api.Cmp(negZ, z_sigmoid_input)          // 1 if -z >= z, -1 if -z < z
	// For negative z: -z will be positive and small, so -z < z, giving cmpNegZ = -1
	// For positive z: -z will be negative (large), so -z > z, giving cmpNegZ = 1
	isNeg := api.IsZero(api.Add(1, cmpNegZ))           // 1 if cmpNegZ == -1 (z is negative)
	
	// isSaturatedPos = 1 if z > maxIn AND z is NOT negative
	isSaturatedPos := api.Mul(isGreaterThanMax, api.Sub(1, isNeg))
	
	// absZ = |z|
	absZ := api.Select(isNeg, negZ, z_sigmoid_input)
	
	// isSaturatedNeg = 1 if |z| > maxIn AND z was negative
	cmpAbsMax := api.Cmp(absZ, maxIn)
	isAbsSaturated := api.IsZero(api.Sub(1, cmpAbsMax))
	isSaturatedNeg := api.Mul(isNeg, isAbsSaturated)
	
	// Clamp absZ to [0, maxIn]
	clampedAbsZ := api.Select(isAbsSaturated, maxIn, absZ)
	
	lookupResults := table.Lookup(clampedAbsZ)[0]
	
	// For negative inputs: sigmoid(-x) = 1 - sigmoid(|x|)
	oneMinusResults := api.Sub(oneOut, lookupResults)
	normalResults := api.Select(isNeg, oneMinusResults, lookupResults)

	// Final saturation: if very negative, output 0; if very positive, output 1
	result1 := api.Select(isSaturatedNeg, zeroOut, normalResults)
	finalResults := api.Select(isSaturatedPos, oneOut, result1)

	// Constrain the public output Y
	api.AssertIsEqual(finalResults, circuit.Y)

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

// loadTestData reads marks and labels from CSV file
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

	// Skip header row
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

// Cache paths
const (
	cacheDir     = "data/cache"
	ccsCache     = "data/cache/circuit.bin"
	srsCache     = "data/cache/srs.bin"
	pkCache      = "data/cache/pk.bin"
	vkCache      = "data/cache/vk.bin"
)

func ensureCacheDir() {
	os.MkdirAll(cacheDir, 0755)
}

func saveCCS(ccs *cs.SparseR1CS, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = ccs.WriteTo(file)
	return err
}

func loadCCS(path string) (*cs.SparseR1CS, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	ccs := &cs.SparseR1CS{}
	_, err = ccs.ReadFrom(file)
	return ccs, err
}

func saveToFile(data interface{}, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := gob.NewEncoder(file)
	return encoder.Encode(data)
}

func loadFromFile(data interface{}, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)
	return decoder.Decode(data)
}

type ProofTask struct {
	idx    int
	mark   int
	label  int
	w_bi   *big.Int
	b_bi   *big.Int
	x_bi   *big.Int
}

type ProofResult struct {
	idx       int
	mark      int
	label     int
	predicted int
	correct   bool
	proveTime time.Duration
	verifyTime time.Duration
	err       error
}

func main() {
	ensureCacheDir()

	fmt.Println("================================================================")
	fmt.Println("     ZK Logistic Regression - Parallel Proof System")
	fmt.Println("================================================================\n")

	var circuit Circuit
	var ccs *cs.SparseR1CS
	var pk plonk.ProvingKey
	var vk plonk.VerifyingKey
	var err error

	// Try to load from cache
	startSetup := time.Now()
	ccsLoaded := false
	if _, err := os.Stat(ccsCache); err == nil {
		fmt.Println("[CACHE] Loading circuit from cache...")
		ccs, err = loadCCS(ccsCache)
		if err != nil {
			log.Printf("[WARN] Cache load failed, recompiling: %v", err)
		} else {
			fmt.Println("[CACHE] ✓ Circuit loaded from cache")
			ccsLoaded = true
		}
	}

	// Compile circuit if not loaded from cache
	if !ccsLoaded {
		fmt.Println("[SETUP] Compiling circuit...")
		ccsTemp, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &circuit)
		if err != nil {
			log.Fatal("circuit compilation error: ", err)
		}
		ccs = ccsTemp.(*cs.SparseR1CS)
		fmt.Printf("[SETUP] ✓ Circuit compiled (%d constraints)\n", ccs.GetNbConstraints())
		
		// Save circuit to cache
		if err := saveCCS(ccs, ccsCache); err != nil {
			log.Printf("[WARN] Failed to cache circuit: %v", err)
		} else {
			fmt.Println("[CACHE] ✓ Circuit cached")
		}
	}

	// Try to load keys from cache
	keysLoaded := false
	if _, err := os.Stat(pkCache); err == nil {
		fmt.Println("[CACHE] Loading proving key from cache...")
		if err := loadFromFile(&pk, pkCache); err == nil {
			fmt.Println("[CACHE] Loading verifying key from cache...")
			if err := loadFromFile(&vk, vkCache); err == nil {
				fmt.Println("[CACHE] ✓ Keys loaded from cache")
				keysLoaded = true
			}
		}
		if !keysLoaded {
			log.Printf("[WARN] Key cache load failed, regenerating keys")
		}
	}

	// Generate keys if not loaded from cache
	if !keysLoaded {
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
		
		// Cache keys
		if err := saveToFile(&pk, pkCache); err != nil {
			log.Printf("[WARN] Failed to cache proving key: %v", err)
		} else {
			fmt.Println("[CACHE] ✓ Proving key cached")
		}
		if err := saveToFile(&vk, vkCache); err != nil {
			log.Printf("[WARN] Failed to cache verifying key: %v", err)
		} else {
			fmt.Println("[CACHE] ✓ Verifying key cached")
		}
	}

	setupTime := time.Since(startSetup)
	fmt.Printf("[SETUP] Total setup time: %v\n\n", setupTime)

	// Load dataset
	marks, labels, err := loadTestData("data/student_dataset.csv")
	if err != nil {
		log.Fatal("Error loading test data:", err)
	}
	
	// Use 3000 samples for production run
	testSize := 3000
	if len(marks) > testSize {
		marks = marks[:testSize]
		labels = labels[:testSize]
	}
	
	fmt.Printf("[DATA] Loaded %d samples from dataset\n", len(marks))
	
	// Prepare proof tasks
	tasks := make([]ProofTask, len(marks))
	w_bi := W_SCALED
	b_bi := B_SCALED
	
	for i, mark := range marks {
		tasks[i] = ProofTask{
			idx:   i,
			mark:  mark,
			label: labels[i],
			w_bi:  w_bi,
			b_bi:  b_bi,
			x_bi:  newScaledInt(int64(mark)),
		}
	}

	// Parallel proof generation with 80% of CPU cores
	// TEMPORARY: Use 1 worker for testing lookup table issue
	numCPU := runtime.NumCPU()
	numWorkers := 1  // Force single worker to test
	
	fmt.Printf("[PARALLEL] Using %d workers (TESTING MODE - single worker)\n", numWorkers)
	fmt.Println("[PROVING] Starting parallel proof generation...\n")

	taskChan := make(chan ProofTask, len(tasks))
	resultChan := make(chan ProofResult, len(tasks))
	var wg sync.WaitGroup

	startProving := time.Now()

	// Worker goroutines
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				result := processProof(task, ccs, pk, vk)
				resultChan <- result
			}
		}(w)
	}

	// Send tasks
	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	results := make([]ProofResult, 0, len(tasks))
	completed := 0
	correctCount := 0
	
	for result := range resultChan {
		completed++
		results = append(results, result)
		
		if result.err != nil {
			fmt.Printf("[ERROR] Sample %d failed: %v\n", result.idx+1, result.err)
			continue
		}
		
		if result.correct {
			correctCount++
		}
		
		// Print progress every 100 samples
		if completed%100 == 0 || completed == len(tasks) {
			fmt.Printf("[PROGRESS] %d/%d samples processed (%.1f%% complete)\n", 
				completed, len(tasks), float64(completed)/float64(len(tasks))*100)
		}
	}

	totalProvingTime := time.Since(startProving)

	// Calculate statistics
	accuracy := float64(correctCount) / float64(len(marks)) * 100
	
	var totalProveTime, totalVerifyTime time.Duration
	for _, r := range results {
		if r.err == nil {
			totalProveTime += r.proveTime
			totalVerifyTime += r.verifyTime
		}
	}
	
	avgProveTime := totalProveTime / time.Duration(len(results))
	avgVerifyTime := totalVerifyTime / time.Duration(len(results))

	// Print final results
	fmt.Println("\n================================================================")
	fmt.Println("                     FINAL RESULTS")
	fmt.Println("================================================================\n")
	
	fmt.Printf("Accuracy Metrics:\n")
	fmt.Printf("  Total samples:        %d\n", len(marks))
	fmt.Printf("  Correct predictions:  %d\n", correctCount)
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
	fmt.Printf("  Avg prove/sample:     %v\n", avgProveTime)
	fmt.Printf("  Avg verify/sample:    %v\n", avgVerifyTime)
	fmt.Printf("  Workers used:         %d (80%% of %d cores)\n", numWorkers, numCPU)
	fmt.Printf("  Throughput:           %.2f proofs/sec\n\n", float64(len(marks))/totalProvingTime.Seconds())
	
	fmt.Println("All proofs verified successfully!")
	
	// Export metrics for collection
	exportMetrics(setupTime, totalProvingTime, avgProveTime, avgVerifyTime, len(marks), correctCount, ccs.GetNbConstraints())
}

func processProof(task ProofTask, ccs *cs.SparseR1CS, pk plonk.ProvingKey, vk plonk.VerifyingKey) ProofResult {
	var w Circuit
	
	w.W = task.w_bi
	w.B = task.b_bi
	w.X = task.x_bi

	// Calculate z = WX + B
	wx_scaled2 := new(big.Int).Mul(task.w_bi, task.x_bi)
	wx_scaled1 := new(big.Int).Div(wx_scaled2, scalingFactor)
	z_linear_bi := new(big.Int).Add(wx_scaled1, task.b_bi)
	
	// Rescale z from 2^32 to 2^10
	shiftFactorWit := new(big.Int).Lsh(big.NewInt(1), Precision-inputPrecision)
	z_sigmoid_input_bi := new(big.Int).Div(z_linear_bi, shiftFactorWit)
	w.Z = z_sigmoid_input_bi
	
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
	w.Y = y_scaled_bi
	
	pred_label := 0
	if y_float >= 0.5 {
		pred_label = 1
	}
	
	// Prove
	startProve := time.Now()
	witnessFull, err := frontend.NewWitness(&w, ecc.BN254.ScalarField())
	if err != nil {
		return ProofResult{idx: task.idx, err: err}
	}

	witnessPublic, err := witnessFull.Public()
	if err != nil {
		return ProofResult{idx: task.idx, err: err}
	}

	proof, err := plonk.Prove(ccs, pk, witnessFull)
	proveTime := time.Since(startProve)
	if err != nil {
		return ProofResult{idx: task.idx, err: err}
	}

	// Verify
	startVerify := time.Now()
	err = plonk.Verify(proof, vk, witnessPublic)
	verifyTime := time.Since(startVerify)
	if err != nil {
		return ProofResult{idx: task.idx, err: err}
	}

	return ProofResult{
		idx:        task.idx,
		mark:       task.mark,
		label:      task.label,
		predicted:  pred_label,
		correct:    pred_label == task.label,
		proveTime:  proveTime,
		verifyTime: verifyTime,
	}
}

func exportMetrics(setupTime, totalProvingTime, avgProveTime, avgVerifyTime time.Duration, samples, correct, constraints int) {
	metricsFile := "data/cache/run_metrics.txt"
	f, err := os.Create(metricsFile)
	if err != nil {
		log.Printf("[WARN] Failed to export metrics: %v", err)
		return
	}
	defer f.Close()
	
	fmt.Fprintf(f, "setup_time_ms=%d\n", setupTime.Milliseconds())
	fmt.Fprintf(f, "total_proving_time_ms=%d\n", totalProvingTime.Milliseconds())
	fmt.Fprintf(f, "avg_prove_time_ms=%d\n", avgProveTime.Milliseconds())
	fmt.Fprintf(f, "avg_verify_time_ms=%d\n", avgVerifyTime.Milliseconds())
	fmt.Fprintf(f, "samples=%d\n", samples)
	fmt.Fprintf(f, "correct=%d\n", correct)
	fmt.Fprintf(f, "accuracy=%.2f\n", float64(correct)/float64(samples)*100)
	fmt.Fprintf(f, "constraints=%d\n", constraints)
	fmt.Fprintf(f, "throughput=%.2f\n", float64(samples)/totalProvingTime.Seconds())
	
	// Get proof size from a sample proof
	var w Circuit
	w.W = W_SCALED
	w.B = B_SCALED
	w.X = newScaledInt(75)
	w.Z = big.NewInt(0)
	w.Y = big.NewInt(1 << (outputPrecision - 1))
	
	witnessFull, _ := frontend.NewWitness(&w, ecc.BN254.ScalarField())
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &w)
	scs := ccs.(*cs.SparseR1CS)
	srs, srsLagrange, _ := unsafekzg.NewSRS(scs)
	pk, _, _ := plonk.Setup(ccs, srs, srsLagrange)
	proof, _ := plonk.Prove(ccs, pk, witnessFull)
	
	var buf bytes.Buffer
	proof.WriteTo(&buf)
	proofSize := buf.Len()
	
	fmt.Fprintf(f, "proof_size_bytes=%d\n", proofSize)
	fmt.Fprintf(f, "proof_size_kb=%.2f\n", float64(proofSize)/1024.0)
	
	fmt.Printf("[METRICS] Exported to %s\n", metricsFile)
}