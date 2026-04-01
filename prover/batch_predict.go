// batch_predict.go — Parallel batch prediction pipeline.
package prover

import (
	"bytes"
	"fmt"
	"io"
	"math/big"
	"runtime"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// keySet holds a private copy of proving/verification keys for one worker.
type keySet struct {
	pk plonk.ProvingKey
	vk plonk.VerifyingKey
}

// cloneKeys creates a deep copy of keys by serializing and deserializing.
func cloneKeys(pk plonk.ProvingKey, vk plonk.VerifyingKey) (*keySet, error) {
	// Serialize PK
	var pkBuf bytes.Buffer
	if _, err := pk.WriteTo(&pkBuf); err != nil {
		return nil, fmt.Errorf("serialize pk: %w", err)
	}
	// Deserialize PK
	newPK := plonk.NewProvingKey(ecc.BN254)
	if _, err := newPK.(io.ReaderFrom).ReadFrom(&pkBuf); err != nil {
		return nil, fmt.Errorf("deserialize pk: %w", err)
	}

	// Serialize VK
	var vkBuf bytes.Buffer
	if _, err := vk.WriteTo(&vkBuf); err != nil {
		return nil, fmt.Errorf("serialize vk: %w", err)
	}
	// Deserialize VK
	newVK := plonk.NewVerifyingKey(ecc.BN254)
	if _, err := newVK.(io.ReaderFrom).ReadFrom(&vkBuf); err != nil {
		return nil, fmt.Errorf("deserialize vk: %w", err)
	}

	return &keySet{pk: newPK, vk: newVK}, nil
}

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

// DefaultKeyPoolSize is the default number of key copies when not specified.
// 16 copies ≈ 530MB memory, good balance for most HPC systems.
const DefaultKeyPoolSize = 16

// BatchPredictParallel runs all predictions using batching + parallelism.
// keyPoolSize controls memory usage: fewer keys = less memory but more serialization.
// Set keyPoolSize=0 for auto (min of numWorkers and DefaultKeyPoolSize).
func BatchPredictParallel(
	setup *BatchSetupResult,
	wFloat []float64, bFloat float64,
	features [][]int,
	numWorkers int,
	keyPoolSize int,
) []*BatchPredResult {
	batchSize := setup.BatchSize

	if numWorkers <= 0 {
		// Use 50% of available cores for HPC optimization
		numWorkers = runtime.NumCPU() / 2
		if numWorkers < 1 {
			numWorkers = 1
		}
	}

	// Determine key pool size (hybrid approach: fewer keys than workers saves memory)
	if keyPoolSize <= 0 {
		// Auto: use min(numWorkers, DefaultKeyPoolSize) for reasonable memory
		keyPoolSize = numWorkers
		if keyPoolSize > DefaultKeyPoolSize {
			keyPoolSize = DefaultKeyPoolSize
		}
	}
	if keyPoolSize > numWorkers {
		keyPoolSize = numWorkers // No point having more keys than workers
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

	// Estimate memory usage
	pkSizeMB := float64(setup.PKSizeBytes) / (1024 * 1024)
	totalMemMB := pkSizeMB * float64(keyPoolSize)

	fmt.Printf("    Batches: %d (size=%d, workers=%d, keys=%d)\n", len(batches), batchSize, numWorkers, keyPoolSize)
	fmt.Printf("    Memory: %.0f MB for key pool (%.1f MB × %d copies)\n", totalMemMB, pkSizeMB, keyPoolSize)

	// Create key pool - hybrid approach allows more workers than keys
	// Workers wait for available keys, enabling memory-efficient parallelism
	keyPool := make(chan *keySet, keyPoolSize)
	for i := 0; i < keyPoolSize; i++ {
		keys, err := cloneKeys(setup.ProvingKey, setup.VerificationKey)
		if err != nil {
			fmt.Printf("    Warning: failed to clone keys %d: %v\n", i, err)
			keys = &keySet{pk: setup.ProvingKey, vk: setup.VerificationKey}
		}
		keyPool <- keys
	}

	results := make([]*BatchPredResult, len(batches))
	var wg sync.WaitGroup
	sem := make(chan struct{}, numWorkers) // Limits concurrent goroutines

	for bIdx, batch := range batches {
		wg.Add(1)
		sem <- struct{}{} // Acquire worker slot

		go func(idx int, batchFeatures [][]int) {
			defer wg.Done()
			defer func() { <-sem }() // Release worker slot

			result := &BatchPredResult{BatchIndex: idx}

			// Witness computation runs in parallel (no key needed yet)
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

			// Create witness (parallel — no key needed)
			witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
			if err != nil {
				result.Error = fmt.Errorf("batch %d witness failed: %w", idx, err)
				results[idx] = result
				return
			}

			// Grab key from pool for proving (may wait if all keys in use)
			keys := <-keyPool
			defer func() { keyPool <- keys }()

			// Prove using worker's private key copy
			proveStart := time.Now()
			proof, err := plonk.Prove(setup.ConstraintSystem, keys.pk, witness)
			result.ProveTime = time.Since(proveStart)

			if err != nil {
				result.Error = fmt.Errorf("batch %d prove failed: %w", idx, err)
				results[idx] = result
				return
			}

			var proofBuf bytes.Buffer
			proof.WriteTo(&proofBuf)
			result.ProofBytes = proofBuf.Bytes()

			// Verify using worker's private key copy
			publicWitness, _ := witness.Public()
			verifyStart := time.Now()
			err = plonk.Verify(proof, keys.vk, publicWitness)
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

