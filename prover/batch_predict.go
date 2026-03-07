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
func ComputeBatchWitness(w1Float, w2Float, bFloat float64, features [][2]int, batchSize int) *circuit.BatchCircuit {
	assignment := circuit.NewBatchCircuit(batchSize)

	scale := circuit.ScalingFactor
	w1Big := new(big.Int).SetInt64(int64(w1Float * float64(scale.Int64())))
	w2Big := new(big.Int).SetInt64(int64(w2Float * float64(scale.Int64())))
	bBig := new(big.Int).SetInt64(int64(bFloat * float64(scale.Int64())))
	assignment.W = [2]frontend.Variable{w1Big, w2Big}
	assignment.B = bBig

	for i := 0; i < batchSize; i++ {
		if i < len(features) {
			single := ComputeWitness(w1Float, w2Float, bFloat, features[i][0], features[i][1])
			assignment.X[i] = [2]frontend.Variable{single.X[0], single.X[1]}
			assignment.Y[i] = single.Y
			assignment.ZTable[i] = single.ZTable
			assignment.Rem[i] = single.Rem
		} else {
			// Pad with dummy prediction
			single := ComputeWitness(w1Float, w2Float, bFloat, 170, 700)
			assignment.X[i] = [2]frontend.Variable{single.X[0], single.X[1]}
			assignment.Y[i] = single.Y
			assignment.ZTable[i] = single.ZTable
			assignment.Rem[i] = single.Rem
		}
	}

	return assignment
}

// BatchPredictParallel runs all predictions using batching + parallelism.
func BatchPredictParallel(
	setup *BatchSetupResult,
	w1Float, w2Float, bFloat float64,
	features [][2]int,
	numWorkers int,
) []*BatchPredResult {
	batchSize := setup.BatchSize

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
		if numWorkers > 8 {
			numWorkers = 8
		}
	}

	// Split features into batches
	var batches [][][2]int
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
	var proveMu sync.Mutex // gnark solver is not fully concurrent-safe

	for bIdx, batch := range batches {
		wg.Add(1)
		sem <- struct{}{}

		go func(idx int, batchFeatures [][2]int) {
			defer wg.Done()
			defer func() { <-sem }()

			result := &BatchPredResult{BatchIndex: idx}

			assignment := ComputeBatchWitness(w1Float, w2Float, bFloat, batchFeatures, batchSize)

			// Build per-prediction results
			for i, f := range batchFeatures {
				prob := float64(assignment.Y[i].(*big.Int).Int64()) / float64(1<<circuit.OutputPrecision)
				pred := "NORMAL"
				if prob >= 0.5 {
					pred = "OVERWEIGHT"
				}
				result.Predictions = append(result.Predictions, &PredictionResult{
					Height:      f[0],
					Weight:      f[1],
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

			// Prove (serialized — gnark solver shares internal state)
			proveMu.Lock()
			proveStart := time.Now()
			proof, err := plonk.Prove(setup.ConstraintSystem, setup.ProvingKey, witness)
			result.ProveTime = time.Since(proveStart)
			proveMu.Unlock()

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
