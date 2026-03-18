// batch_predict.go — Parallel batch prediction pipeline.
package prover

import (
	"bytes"
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// BatchPredResult holds results for one batch of predictions.
type BatchPredResult struct {
	BatchIndex  int
	Predictions []*PredictionResult
	ProveTime   time.Duration
	VerifyTime  time.Duration
	ProofBytes  []byte
	Verified    bool
	Error       error
}

// ComputeBatchWitness fills a BatchCircuit assignment for the given features.
func ComputeBatchWitness(wFloat []float64, bFloat float64, features [][]int, batchSize int) *circuit.BatchCircuit {
	numFeatures := len(wFloat)
	assignment := circuit.NewBatchCircuit(batchSize, numFeatures)

	scale := circuit.ScalingFactor
	
	wBig := make([]frontend.Variable, numFeatures)
	for i := range wFloat {
		wBig[i] = new(big.Int).SetInt64(int64(wFloat[i] * float64(scale.Int64())))
	}
	
	bBig := new(big.Int).SetInt64(int64(bFloat * float64(scale.Int64())))
	assignment.W = wBig
	assignment.B = bBig

	// Setup dummy features for padding
	dummyFeatures := make([]int, numFeatures)
	for i := range dummyFeatures {
		dummyFeatures[i] = 100 // Safe dummy value
	}

	for i := 0; i < batchSize; i++ {
		var single *circuit.LRCircuit
		if i < len(features) {
			single = ComputeWitness(wFloat, bFloat, features[i])
		} else {
			// Pad with dummy prediction
			single = ComputeWitness(wFloat, bFloat, dummyFeatures)
		}
		
		for j := 0; j < numFeatures; j++ {
			assignment.X[i][j] = single.X[j]
		}
		assignment.Y[i] = single.Y
		assignment.ZTable[i] = single.ZTable
		assignment.Rem[i] = single.Rem
	}

	return assignment
}

// BatchPredictParallel runs all predictions using batching + parallelism.
func BatchPredictParallel(
	setup *BatchSetupResult,
	wFloat []float64, bFloat float64,
	features [][]int,
	numWorkers int,
) []*BatchPredResult {
	batchSize := setup.BatchSize

	if numWorkers <= 0 {
		// Use 50% of available cores for HPC optimization
		numWorkers = runtime.NumCPU() / 2
		if numWorkers < 1 {
			numWorkers = 1
		}
	}

	// Split features into batches
	var batches [][][]int
	for i := 0; i < len(features); i += batchSize {
		end := i + batchSize
		if end > len(features) {
			end = len(features)
		}
		batches = append(batches, features[i:end])
	}

	fmt.Printf("    Batches: %d (size=%d, workers=%d)\n", len(batches), batchSize, numWorkers)

	results := make([]*BatchPredResult, len(batches))
	var wg sync.WaitGroup
	sem := make(chan struct{}, numWorkers)
	// Note: gnark v0.11+ is thread-safe, mutex removed for parallel proving

	for bIdx, batch := range batches {
		wg.Add(1)
		sem <- struct{}{}

		go func(idx int, batchFeatures [][]int) {
			defer wg.Done()
			defer func() { <-sem }()

			result := &BatchPredResult{BatchIndex: idx}

			assignment := ComputeBatchWitness(wFloat, bFloat, batchFeatures, batchSize)

			// Build per-prediction results
			for i, f := range batchFeatures {
				prob := float64(assignment.Y[i].(*big.Int).Int64()) / float64(1<<circuit.OutputPrecision)
				pred := "NORMAL"
				if prob >= 0.5 {
					pred = "OVERWEIGHT"
				}
				
				// Grab first two features for log formatting if they exist, otherwise pad 0
				h := 0
				w := 0
				if len(f) > 0 { h = f[0] }
				if len(f) > 1 { w = f[1] }
				
				result.Predictions = append(result.Predictions, &PredictionResult{
					Height:      h,
					Weight:      w,
					Probability: prob,
					Prediction:  pred,
				})
			}

			// Create witness (parallel — this is safe)
			witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
			if err != nil {
				result.Error = fmt.Errorf("batch %d witness failed: %w", idx, err)
				results[idx] = result
				return
			}

			// Prove (parallel - gnark v0.11 is thread-safe)
			proveStart := time.Now()
			proof, err := plonk.Prove(setup.ConstraintSystem, setup.ProvingKey, witness)
			result.ProveTime = time.Since(proveStart)

			if err != nil {
				result.Error = fmt.Errorf("batch %d prove failed: %w", idx, err)
				results[idx] = result
				return
			}

			var proofBuf bytes.Buffer
			proof.WriteTo(&proofBuf)
			result.ProofBytes = proofBuf.Bytes()

			// Verify (parallel — this is safe)
			publicWitness, _ := witness.Public()
			verifyStart := time.Now()
			err = plonk.Verify(proof, setup.VerificationKey, publicWitness)
			result.VerifyTime = time.Since(verifyStart)
			result.Verified = (err == nil)

			if err != nil {
				result.Error = fmt.Errorf("batch %d verify failed: %w", idx, err)
			}

			results[idx] = result
		}(bIdx, batch)
	}

	wg.Wait()
	return results
}
