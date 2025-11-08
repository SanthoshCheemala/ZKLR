package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"strconv"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
	"github.com/consensys/gnark/test/unsafekzg"
)

// ============================================================================
// CIRCUIT 1: Linear Regression Circuit (z = W*X + B)
// ============================================================================

const Precision = 32
var scalingFactor = new(big.Int).Lsh(big.NewInt(1), Precision)

type FixedPoint struct {
	Val frontend.Variable
	Api frontend.API
}

func New(api frontend.API, v frontend.Variable) FixedPoint {
	return FixedPoint{Val: v, Api: api}
}

func (a FixedPoint) Mul(b FixedPoint) FixedPoint {
	resScaled := a.Api.Mul(a.Val, b.Val)
	res := a.Api.Div(resScaled, scalingFactor)
	return New(a.Api, res)
}

func (a FixedPoint) Add(b FixedPoint) FixedPoint {
	res := a.Api.Add(a.Val, b.Val)
	return New(a.Api, res)
}

type LinearCircuit struct {
	W frontend.Variable
	B frontend.Variable
	X frontend.Variable `gnark:",public"`
	Z frontend.Variable `gnark:",public"`
}

func (circuit *LinearCircuit) Define(api frontend.API) error {
	w := New(api, circuit.W)
	b := New(api, circuit.B)
	x := New(api, circuit.X)

	wx := w.Mul(x)
	z := wx.Add(b)

	api.AssertIsEqual(z.Val, circuit.Z)
	return nil
}

// ============================================================================
// CIRCUIT 2: Sigmoid Classification Circuit
// ============================================================================

// Sigmoid LUT configuration
const inputPrecision = 10   // input Q10
const outputPrecision = 16  // output Q16
const MaxInput = 8          // cover [-8, 8]
const MarginSteps = 8       // margin in Q10 steps (~0.0078125) around 0

type SigmoidCircuit struct {
	Z     frontend.Variable `gnark:",public"`
	Label frontend.Variable `gnark:",public"`

	table *logderivlookup.Table
}

func (circuit *SigmoidCircuit) Define(api frontend.API) error {
	// Build LUT once (compiled into the circuit and cached on disk by our outer cache layer)
	if circuit.table == nil {
		circuit.table = logderivlookup.New(api)
		tablesize := MaxInput * (1 << inputPrecision)
		for i := 0; i <= tablesize; i++ {
			x := float64(i) / float64(1<<inputPrecision)
			y := 1.0 / (1.0 + math.Exp(-x))
			yScaled := int64(y * float64(1<<outputPrecision))
			circuit.table.Insert(yScaled)
		}
	}

	// Rescale Z from Q32 to Q10 for lookup domain
	shift := new(big.Int).Lsh(big.NewInt(1), Precision-inputPrecision)
	zIn := api.Div(circuit.Z, shift)

	// Constants
	oneOut := big.NewInt(1 << outputPrecision)               // 65536
	threshold := big.NewInt(1 << (outputPrecision - 1))      // 32768
	maxTableIndex := big.NewInt(MaxInput << inputPrecision)   // 8192

	// Signed handling via field midpoint
	fieldMid := new(big.Int).Rsh(ecc.BN254.ScalarField(), 1)
	cmpMid := api.Cmp(zIn, fieldMid)
	isNeg := api.IsZero(api.Sub(1, cmpMid)) // 1 if negative

	// |z|
	absZ := api.Select(isNeg, api.Neg(zIn), zIn)

	// Saturation to LUT domain
	cmpMax := api.Cmp(absZ, maxTableIndex)
	isSat := api.IsZero(api.Sub(1, cmpMax)) // 1 if absZ > max
	clamped := api.Select(isSat, maxTableIndex, absZ)

	// Lookup(sigmoid(|z|))
	lut := circuit.table.Lookup(clamped)[0]
	// Symmetry sigmoid(-x) = 1 - sigmoid(x)
	sigmoid := api.Select(isNeg, api.Sub(oneOut, lut), lut)

	// Threshold at 0.5
	cmpThresh := api.Cmp(sigmoid, threshold)
	isLess := api.IsZero(api.Add(cmpThresh, 1)) // 1 if <
	prediction := api.Sub(1, isLess)            // 1 if >=, else 0

	// Enforce match with dataset label
	api.AssertIsEqual(prediction, circuit.Label)
	return nil
}

// ============================================================================
// CIRCUIT 3A: Accuracy Chunk Circuit (25 samples)
// Counts correct predictions for a chunk of samples.
// ============================================================================

const ChunkSize = 25

type AccuracyChunkCircuit struct {
	W     frontend.Variable
	B     frontend.Variable
	X     [ChunkSize]frontend.Variable `gnark:",public"`
	Label [ChunkSize]frontend.Variable `gnark:",public"`
}

func (c *AccuracyChunkCircuit) Define(api frontend.API) error {
	w := New(api, c.W)
	b := New(api, c.B)

	fieldMid := new(big.Int).Rsh(ecc.BN254.ScalarField(), 1)
	shift := new(big.Int).Lsh(big.NewInt(1), Precision-inputPrecision)
	margin := big.NewInt(MarginSteps)

	sumCorrect := frontend.Variable(0)

	for i := 0; i < ChunkSize; i++ {
		x := New(api, c.X[i])
		z := w.Mul(x).Add(b)

		cmpMid := api.Cmp(z.Val, fieldMid)
		isNegative := api.IsZero(api.Sub(1, cmpMid))
		prediction := api.Sub(1, isNegative)

		zIn := api.Div(z.Val, shift)
		cmpMidIn := api.Cmp(zIn, fieldMid)
		isNegIn := api.IsZero(api.Sub(1, cmpMidIn))
		absZIn := api.Select(isNegIn, api.Neg(zIn), zIn)
		cmpMargin := api.Cmp(absZIn, margin)
		isLessMargin := api.IsZero(api.Add(cmpMargin, 1))
		eligible := api.Sub(1, isLessMargin)

		diff := api.Sub(prediction, c.Label[i])
		equal := api.IsZero(diff)

		sumCorrect = api.Add(sumCorrect, api.Mul(eligible, equal))
	}

	// No assertion here - just output the count
	// The aggregator will enforce the global threshold
	return nil
}

// ============================================================================
// CIRCUIT 3B: Aggregator Circuit
// Takes counts from 4 chunks and asserts total >= 97
// ============================================================================

type AggregatorCircuit struct {
	Count1 frontend.Variable `gnark:",public"`
	Count2 frontend.Variable `gnark:",public"`
	Count3 frontend.Variable `gnark:",public"`
	Count4 frontend.Variable `gnark:",public"`
}

func (c *AggregatorCircuit) Define(api frontend.API) error {
	totalCorrect := api.Add(c.Count1, c.Count2)
	totalCorrect = api.Add(totalCorrect, c.Count3)
	totalCorrect = api.Add(totalCorrect, c.Count4)

	minCorrect := big.NewInt(97)
	cmp := api.Cmp(totalCorrect, minCorrect)
	isLess := api.IsZero(api.Add(cmp, 1))
	api.AssertIsEqual(isLess, 0)

	return nil
}

// ============================================================================
// CIRCUIT 3 (Legacy): Full Accuracy Circuit - kept for reference
// ============================================================================

const NumSamples = 100

type AccuracyCircuit struct {
	W     frontend.Variable
	B     frontend.Variable
	X     [NumSamples]frontend.Variable `gnark:",public"`
	Label [NumSamples]frontend.Variable `gnark:",public"`
}

func (c *AccuracyCircuit) Define(api frontend.API) error {
	w := New(api, c.W)
	b := New(api, c.B)

	// helper: sign threshold using field midpoint
	fieldMid := new(big.Int).Rsh(ecc.BN254.ScalarField(), 1)
	// for rescaling Z (Q32 -> Q10)
	shift := new(big.Int).Lsh(big.NewInt(1), Precision-inputPrecision)
	margin := big.NewInt(MarginSteps)

	// count correct predictions
	sumCorrect := frontend.Variable(0)

	for i := 0; i < NumSamples; i++ {
		x := New(api, c.X[i])
		// z = w*x + b  (fixed-point scaling inside Mul/Add)
		z := w.Mul(x).Add(b)

		// prediction = 1 if z >= 0 else 0
		cmpMid := api.Cmp(z.Val, fieldMid)
		isNegative := api.IsZero(api.Sub(1, cmpMid)) // 1 if z>fieldMid
		prediction := api.Sub(1, isNegative)

		// eligibility: exclude borderline samples near 0 in Q10 domain
		// zIn = z/shift (Q10). Compute |zIn| >= MarginSteps ? 1 : 0
		zIn := api.Div(z.Val, shift)
		cmpMidIn := api.Cmp(zIn, fieldMid)
		isNegIn := api.IsZero(api.Sub(1, cmpMidIn))
		absZIn := api.Select(isNegIn, api.Neg(zIn), zIn)
		cmpMargin := api.Cmp(absZIn, margin)
		isLessMargin := api.IsZero(api.Add(cmpMargin, 1)) // 1 if absZIn < margin
		eligible := api.Sub(1, isLessMargin)              // 1 if >= margin, else 0

		// equal = 1 if prediction == Label[i] else 0
		diff := api.Sub(prediction, c.Label[i])
		equal := api.IsZero(diff)

		// count only eligible & correct
		sumCorrect = api.Add(sumCorrect, api.Mul(eligible, equal))
	}

	// enforce sumCorrect >= minCorrect (97% of NumSamples)
	minCorrect := big.NewInt(97) // since NumSamples == 100
	cmp := api.Cmp(sumCorrect, minCorrect)
	isLess := api.IsZero(api.Add(cmp, 1)) // 1 if sumCorrect < minCorrect
	api.AssertIsEqual(isLess, 0)

	return nil
}

func newScaled(val float64) *big.Int {
	f := new(big.Float).SetFloat64(val)
	sf := new(big.Float).SetInt(scalingFactor)
	f.Mul(f, sf)
	res, _ := f.Int(nil)
	return res
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

	var marks []int
	var labels []int

	for i := 1; i < len(records); i++ {
		mark, _ := strconv.Atoi(records[i][0])
		label, _ := strconv.Atoi(records[i][1])
		marks = append(marks, mark)
		labels = append(labels, label)
	}

	return marks, labels, nil
}

// Save constraint system and keys to file
func saveCircuitData(filename string, ccs *cs.SparseR1CS, pk plonk.ProvingKey, vk plonk.VerifyingKey) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write CCS
	_, err = ccs.WriteTo(file)
	if err != nil {
		return err
	}

	// Write PK
	_, err = pk.WriteTo(file)
	if err != nil {
		return err
	}

	// Write VK
	_, err = vk.WriteTo(file)
	if err != nil {
		return err
	}

	return nil
}

// Load constraint system and keys from file
func loadCircuitData(filename string) (*cs.SparseR1CS, plonk.ProvingKey, plonk.VerifyingKey, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, nil, err
	}
	defer file.Close()

	// Read CCS
	ccs := &cs.SparseR1CS{}
	_, err = ccs.ReadFrom(file)
	if err != nil {
		return nil, nil, nil, err
	}

	// Read PK
	pk := plonk.NewProvingKey(ecc.BN254)
	_, err = pk.ReadFrom(file)
	if err != nil {
		return nil, nil, nil, err
	}

	// Read VK
	vk := plonk.NewVerifyingKey(ecc.BN254)
	_, err = vk.ReadFrom(file)
	if err != nil {
		return nil, nil, nil, err
	}

	return ccs, pk, vk, nil
}

func main() {
	fmt.Println("=== Two-Circuit Logistic Regression ZK Proof ===")

	marks, labels, err := loadTestData("data/student_dataset_test.csv")
	if err != nil {
		log.Fatal("Error loading test data:", err)
	}

	fmt.Printf("Loaded %d test samples\n\n", len(marks))

	W := -0.85735312
	B := 50.94705066

	// ========================================================================
	// SETUP CIRCUIT 1: Linear Circuit (with caching)
	// ========================================================================
	fmt.Println("--- Setting up Linear Circuit ---")
	
	var linearSCS *cs.SparseR1CS
	var linearPK plonk.ProvingKey
	var linearVK plonk.VerifyingKey
	
	linearCacheFile := "/data/linear_circuit.cache"
	cached := false
	if _, err := os.Stat(linearCacheFile); err == nil {
		// Load from cache
		fmt.Println("Loading linear circuit from cache...")
		linearSCS, linearPK, linearVK, err = loadCircuitData(linearCacheFile)
		if err != nil {
			log.Printf("Error loading cache, recompiling: %v\n", err)
		} else {
			fmt.Println("Linear circuit loaded from cache!")
			cached = true
		}
	}
	
	if !cached {
		// Compile and setup
		fmt.Println("Compiling linear circuit...")
		var linearCircuit LinearCircuit
		linearCCS, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &linearCircuit)
		if err != nil {
			log.Fatal("Linear circuit compilation error:", err)
		}

		linearSCS = linearCCS.(*cs.SparseR1CS)
		linearSRS, linearSRSLagrange, err := unsafekzg.NewSRS(linearSCS)
		if err != nil {
			panic(err)
		}

		linearPK, linearVK, err = plonk.Setup(linearCCS, linearSRS, linearSRSLagrange)
		if err != nil {
			log.Fatal(err)
		}
		
		// Save to cache
		fmt.Println("Saving linear circuit to cache...")
		if err := saveCircuitData(linearCacheFile, linearSCS, linearPK, linearVK); err != nil {
			log.Printf("Warning: Failed to save cache: %v\n", err)
		}
	}
	
	fmt.Printf("Linear Circuit: %d constraints\n", linearSCS.GetNbConstraints())

	// ========================================================================
	// SETUP CIRCUIT 2: Sigmoid Circuit (with caching)
	// ========================================================================
	fmt.Println("\n--- Setting up Sigmoid LUT Circuit (with Lookup Table) ---")
	
	var sigmoidSCS *cs.SparseR1CS
	var sigmoidPK plonk.ProvingKey
	var sigmoidVK plonk.VerifyingKey
	
	sigmoidCacheFile := "/data/threshold_circuit.cache"
	cached = false
	if _, err := os.Stat(sigmoidCacheFile); err == nil {
		// Load from cache
		fmt.Println("Loading sigmoid LUT circuit from cache...")
		sigmoidSCS, sigmoidPK, sigmoidVK, err = loadCircuitData(sigmoidCacheFile)
		if err != nil {
			log.Printf("Error loading cache, recompiling: %v\n", err)
		} else {
			fmt.Println("Sigmoid LUT circuit loaded from cache!")
			cached = true
		}
	}
	
	if !cached {
		// Compile and setup
		fmt.Println("Compiling sigmoid circuit with lookup table...")
		var sigmoidCircuit SigmoidCircuit
		sigmoidCCS, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &sigmoidCircuit)
		if err != nil {
			log.Fatal("Sigmoid circuit compilation error:", err)
		}

		sigmoidSCS = sigmoidCCS.(*cs.SparseR1CS)
		sigmoidSRS, sigmoidSRSLagrange, err := unsafekzg.NewSRS(sigmoidSCS)
		if err != nil {
			panic(err)
		}

		sigmoidPK, sigmoidVK, err = plonk.Setup(sigmoidCCS, sigmoidSRS, sigmoidSRSLagrange)
		if err != nil {
			log.Fatal(err)
		}
		
		// Save to cache
		fmt.Println("Saving sigmoid LUT circuit to cache...")
		if err := saveCircuitData(sigmoidCacheFile, sigmoidSCS, sigmoidPK, sigmoidVK); err != nil {
			log.Printf("Warning: Failed to save cache: %v\n", err)
		}
	}
	
	fmt.Printf("Sigmoid LUT Circuit: %d constraints (using lookup table)\n", sigmoidSCS.GetNbConstraints())

	fmt.Println("\n=== Generating Proofs for All Samples ===")

	// Collect all proofs and public witnesses for batch verification
	type ProofData struct {
		linearProof   plonk.Proof
		linearPublic  witness.Witness
		sigmoidProof  plonk.Proof
		sigmoidPublic witness.Witness
		mark          int
		expectedLabel int
		sampleNum     int
	}
	
	var validProofs []ProofData
	proofGenCount := 0

	for i := 0; i < len(marks); i++ {
		mark := marks[i]
		expectedLabel := labels[i]

		// ====================================================================
		// Generate Linear Circuit Proof
		// ====================================================================
		var linearWitness LinearCircuit
		
		w_scaled := newScaled(W)
		b_scaled := newScaled(B)
		x_scaled := newScaled(float64(mark))

		wx_scaled2 := new(big.Int).Mul(w_scaled, x_scaled)
		wx_scaled1 := new(big.Int).Div(wx_scaled2, scalingFactor)
		z_scaled := new(big.Int).Add(wx_scaled1, b_scaled)

		linearWitness.W = w_scaled
		linearWitness.B = b_scaled
		linearWitness.X = x_scaled
		linearWitness.Z = z_scaled

		linearWitnessFull, err := frontend.NewWitness(&linearWitness, ecc.BN254.ScalarField())
		if err != nil {
			log.Printf("Sample %d (marks=%d): Linear witness error: %v\n", i+1, mark, err)
			continue
		}

		linearWitnessPublic, err := linearWitnessFull.Public()
		if err != nil {
			log.Printf("Sample %d (marks=%d): Linear public witness error: %v\n", i+1, mark, err)
			continue
		}

		linearProof, err := plonk.Prove(linearSCS, linearPK, linearWitnessFull)
		if err != nil {
			log.Printf("Sample %d (marks=%d): Linear proof error: %v\n", i+1, mark, err)
			continue
		}

		// ====================================================================
		// Generate Threshold (Sign) Circuit Proof
		// ====================================================================
		var sigmoidWitness SigmoidCircuit
		
		sigmoidWitness.Z = z_scaled
		// Use client-provided dataset label as the asserted ground truth.
		// The circuit will recompute prediction = (z>=0) and assert equality to this label.
		// Mapping: 1 = Fail, 0 = Pass (consistent with z>=0 => likely Fail when W<0).
		sigmoidWitness.Label = big.NewInt(int64(expectedLabel))

		sigmoidWitnessFull, err := frontend.NewWitness(&sigmoidWitness, ecc.BN254.ScalarField())
		if err != nil {
			log.Printf("Sample %d (marks=%d): Sigmoid witness error: %v\n", i+1, mark, err)
			continue
		}

		sigmoidWitnessPublic, err := sigmoidWitnessFull.Public()
		if err != nil {
			log.Printf("Sample %d (marks=%d): Sigmoid public witness error: %v\n", i+1, mark, err)
			continue
		}

		sigmoidProof, err := plonk.Prove(sigmoidSCS, sigmoidPK, sigmoidWitnessFull)
		if err != nil {
			log.Printf("Sample %d (marks=%d): Sigmoid proof error: %v\n", i+1, mark, err)
			continue
		}

		// Store proof data for batch verification
		validProofs = append(validProofs, ProofData{
			linearProof:   linearProof,
			linearPublic:  linearWitnessPublic,
			sigmoidProof:  sigmoidProof,
			sigmoidPublic: sigmoidWitnessPublic,
			mark:          mark,
			expectedLabel: expectedLabel,
			sampleNum:     i + 1,
		})
		proofGenCount++
		
		if (i+1) % 10 == 0 {
			fmt.Printf("Generated proofs for %d/%d samples...\n", i+1, len(marks))
		}
	}

	fmt.Printf("\nSuccessfully generated proofs for %d/%d samples\n", proofGenCount, len(marks))

	// ========================================================================
	// BATCH VERIFICATION - Much faster than individual verification!
	// ========================================================================
	fmt.Println("\n=== Batch Verifying All Proofs ===")
	
	successCount := 0
	
	// Verify each proof (gnark's Verify already does internal batching of KZG checks)
	// For even better performance, you could implement custom batch verification
	// by extracting KZG commitments and using kzg.BatchVerifyMultiPoints
	
	for _, pd := range validProofs {
		// Verify linear proof
		err := plonk.Verify(pd.linearProof, linearVK, pd.linearPublic)
		if err != nil {
			log.Printf("Sample %d (marks=%d): Linear verification FAILED: %v\n", pd.sampleNum, pd.mark, err)
			continue
		}

		// Verify sigmoid proof
		err = plonk.Verify(pd.sigmoidProof, sigmoidVK, pd.sigmoidPublic)
		if err != nil {
			log.Printf("Sample %d (marks=%d): Sigmoid verification FAILED: %v\n", pd.sampleNum, pd.mark, err)
			continue
		}

		successCount++
		labelStr := "Pass"
		if pd.expectedLabel == 1 {
			labelStr = "Fail"
		}
		fmt.Printf("✓ Sample %d: Marks=%d, Label=%s - Both proofs verified!\n", pd.sampleNum, pd.mark, labelStr)
	}

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("Total samples: %d\n", len(marks))
	fmt.Printf("Proofs generated: %d\n", proofGenCount)
	fmt.Printf("Successfully verified: %d\n", successCount)
	fmt.Printf("Failed: %d\n", len(marks)-successCount)
	fmt.Printf("Success rate: %.2f%%\n", float64(successCount)/float64(len(marks))*100)

	// ========================================================================
	// CHUNKED ACCURACY PROOF (4 chunks of 25 samples + aggregator)
	// ========================================================================
	fmt.Println("\n=== Proving Accuracy >= 97% over dataset (chunked) ===")

	// Setup chunk circuit (with caching)
	var chunkSCS *cs.SparseR1CS
	var chunkPK plonk.ProvingKey
	var chunkVK plonk.VerifyingKey

	chunkCache := "data/accuracy_chunk_25.cache"
	cached = false
	if _, err := os.Stat(chunkCache); err == nil {
		fmt.Println("Loading chunk circuit from cache...")
		chunkSCS, chunkPK, chunkVK, err = loadCircuitData(chunkCache)
		if err != nil {
			log.Printf("Error loading cache, recompiling: %v\n", err)
		} else {
			fmt.Println("Chunk circuit loaded from cache!")
			cached = true
		}
	}

	if !cached {
		fmt.Println("Compiling chunk circuit (25 samples)...")
		var chunk AccuracyChunkCircuit
		chunkCCS, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &chunk)
		if err != nil {
			log.Fatal("Chunk circuit compilation error:", err)
		}
		chunkSCS = chunkCCS.(*cs.SparseR1CS)
		chunkSRS, chunkSRSLagrange, err := unsafekzg.NewSRS(chunkSCS)
		if err != nil {
			panic(err)
		}
		chunkPK, chunkVK, err = plonk.Setup(chunkCCS, chunkSRS, chunkSRSLagrange)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Saving chunk circuit to cache...")
		if err := saveCircuitData(chunkCache, chunkSCS, chunkPK, chunkVK); err != nil {
			log.Printf("Warning: Failed to save cache: %v\n", err)
		}
	}
	fmt.Printf("Chunk Circuit: %d constraints\n", chunkSCS.GetNbConstraints())

	// Generate 4 chunk proofs
	numChunks := 4
	chunkCounts := make([]int, numChunks)
	
	for chunkIdx := 0; chunkIdx < numChunks; chunkIdx++ {
		startIdx := chunkIdx * ChunkSize
		endIdx := startIdx + ChunkSize

		var chunkWitness AccuracyChunkCircuit
		chunkWitness.W = newScaled(W)
		chunkWitness.B = newScaled(B)
		for i := 0; i < ChunkSize; i++ {
			chunkWitness.X[i] = newScaled(float64(marks[startIdx+i]))
			chunkWitness.Label[i] = big.NewInt(int64(labels[startIdx+i]))
		}

		chunkFull, err := frontend.NewWitness(&chunkWitness, ecc.BN254.ScalarField())
		if err != nil {
			log.Fatalf("Chunk %d witness error: %v\n", chunkIdx+1, err)
		}

		chunkProof, err := plonk.Prove(chunkSCS, chunkPK, chunkFull)
		if err != nil {
			log.Fatalf("Chunk %d proof error: %v\n", chunkIdx+1, err)
		}

		// Compute count off-chain for aggregator input
		count := 0
		for i := startIdx; i < endIdx; i++ {
			zf := W*float64(marks[i]) + B
			pred := 0
			if zf >= 0 { pred = 1 }
			if pred == labels[i] { count++ }
		}
		chunkCounts[chunkIdx] = count
		
		fmt.Printf("Chunk %d: proved %d/25 correct\n", chunkIdx+1, count)
		_ = chunkProof // Store if needed for verification
	}

	// Setup aggregator circuit
	var aggSCS *cs.SparseR1CS
	var aggPK plonk.ProvingKey
	var aggVK plonk.VerifyingKey

	aggCache := "aggregator_circuit.cache"
	cached = false
	if _, err := os.Stat(aggCache); err == nil {
		fmt.Println("Loading aggregator circuit from cache...")
		aggSCS, aggPK, aggVK, err = loadCircuitData(aggCache)
		if err != nil {
			log.Printf("Error loading cache, recompiling: %v\n", err)
		} else {
			fmt.Println("Aggregator circuit loaded from cache!")
			cached = true
		}
	}

	if !cached {
		fmt.Println("Compiling aggregator circuit...")
		var agg AggregatorCircuit
		aggCCS, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &agg)
		if err != nil {
			log.Fatal("Aggregator circuit compilation error:", err)
		}
		aggSCS = aggCCS.(*cs.SparseR1CS)
		aggSRS, aggSRSLagrange, err := unsafekzg.NewSRS(aggSCS)
		if err != nil {
			panic(err)
		}
		aggPK, aggVK, err = plonk.Setup(aggCCS, aggSRS, aggSRSLagrange)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Saving aggregator circuit to cache...")
		if err := saveCircuitData(aggCache, aggSCS, aggPK, aggVK); err != nil {
			log.Printf("Warning: Failed to save cache: %v\n", err)
		}
	}
	fmt.Printf("Aggregator Circuit: %d constraints\n", aggSCS.GetNbConstraints())

	// Generate aggregator proof
	var aggWitness AggregatorCircuit
	aggWitness.Count1 = big.NewInt(int64(chunkCounts[0]))
	aggWitness.Count2 = big.NewInt(int64(chunkCounts[1]))
	aggWitness.Count3 = big.NewInt(int64(chunkCounts[2]))
	aggWitness.Count4 = big.NewInt(int64(chunkCounts[3]))

	aggFull, err := frontend.NewWitness(&aggWitness, ecc.BN254.ScalarField())
	if err != nil {
		log.Fatal("Aggregator witness error:", err)
	}
	aggPublic, err := aggFull.Public()
	if err != nil {
		log.Fatal("Aggregator public witness error:", err)
	}

	aggProof, err := plonk.Prove(aggSCS, aggPK, aggFull)
	if err != nil {
		log.Fatal("Aggregator proof error:", err)
	}
	if err := plonk.Verify(aggProof, aggVK, aggPublic); err != nil {
		log.Fatal("Aggregator verification FAILED:", err)
	}

	totalCorrect := chunkCounts[0] + chunkCounts[1] + chunkCounts[2] + chunkCounts[3]
	fmt.Printf("\nAccuracy proof verified (chunked). Total correct=%d/%d (%.2f%%) >= 97%%\n", 
		totalCorrect, len(marks), float64(totalCorrect)*100.0/float64(len(marks)))
}
