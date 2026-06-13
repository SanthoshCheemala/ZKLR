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
	"github.com/consensys/gnark/constraint"
	csbn254 "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// keySet holds a private copy of proving/verification keys and constraint
// system for one worker. All three must be per-worker: gnark's cs.Solve()
// (called inside plonk.Prove) mutates the logderivlookup blueprint state,
// so sharing a CS across concurrent goroutines corrupts the sigmoid table.
type keySet struct {
	pk plonk.ProvingKey
	vk plonk.VerifyingKey
	cs constraint.ConstraintSystem
}

// cloneKeys creates a deep copy of pk, vk, and cs by serializing and
// deserializing each. This is the only safe way to get independent CS copies
// because gnark's in-memory Clone does not deep-copy blueprint table data.
func cloneKeys(pk plonk.ProvingKey, vk plonk.VerifyingKey, cs constraint.ConstraintSystem) (*keySet, error) {
	var pkBuf bytes.Buffer
	if _, err := pk.WriteTo(&pkBuf); err != nil {
		return nil, fmt.Errorf("serialize pk: %w", err)
	}
	newPK := plonk.NewProvingKey(ecc.BN254)
	if _, err := newPK.(io.ReaderFrom).ReadFrom(&pkBuf); err != nil {
		return nil, fmt.Errorf("deserialize pk: %w", err)
	}

	var vkBuf bytes.Buffer
	if _, err := vk.WriteTo(&vkBuf); err != nil {
		return nil, fmt.Errorf("serialize vk: %w", err)
	}
	newVK := plonk.NewVerifyingKey(ecc.BN254)
	if _, err := newVK.(io.ReaderFrom).ReadFrom(&vkBuf); err != nil {
		return nil, fmt.Errorf("deserialize vk: %w", err)
	}

	var csBuf bytes.Buffer
	if _, err := cs.WriteTo(&csBuf); err != nil {
		return nil, fmt.Errorf("serialize cs: %w", err)
	}
	newCS := new(csbn254.SparseR1CS)
	if _, err := newCS.ReadFrom(&csBuf); err != nil {
		return nil, fmt.Errorf("deserialize cs: %w", err)
	}

	return &keySet{pk: newPK, vk: newVK, cs: newCS}, nil
}

// BatchPredResult holds results for one batch of predictions.
type BatchPredResult struct {
	BatchIndex  int
	Predictions []*PredictionResult
	ProveTime   time.Duration
	VerifyTime  time.Duration
	ProofBytes  []byte
	// PublicWitnessBytes is the serialized public witness (X, outputs,
	// commitment) — together with the VK and proof this is everything an
	// independent verifier needs.
	PublicWitnessBytes []byte
	Verified           bool
	Error              error
}

// computeBatchSingles computes per-sample witnesses, padding the batch with a
// safe dummy sample so the assignment always matches the compiled batch size.
func computeBatchSingles(wFloat []float64, bFloat float64, features [][]int, batchSize int) []*circuit.LRCircuit {
	numFeatures := len(wFloat)
	dummyFeatures := make([]int, numFeatures)
	for i := range dummyFeatures {
		dummyFeatures[i] = 100 // Safe dummy value
	}

	singles := make([]*circuit.LRCircuit, batchSize)
	for i := 0; i < batchSize; i++ {
		if i < len(features) {
			singles[i] = ComputeWitness(wFloat, bFloat, features[i])
		} else {
			singles[i] = ComputeWitness(wFloat, bFloat, dummyFeatures)
		}
	}
	return singles
}

// ComputeBatchWitness fills a BatchCircuit (full-probability) assignment.
func ComputeBatchWitness(wFloat []float64, bFloat float64, features [][]int, batchSize int) *circuit.BatchCircuit {
	numFeatures := len(wFloat)
	singles := computeBatchSingles(wFloat, bFloat, features, batchSize)

	assignment := circuit.NewBatchCircuit(batchSize, numFeatures)
	assignment.W = singles[0].W
	assignment.B = singles[0].B
	assignment.Commitment = singles[0].Commitment

	for i := 0; i < batchSize; i++ {
		for j := 0; j < numFeatures; j++ {
			assignment.X[i][j] = singles[i].X[j]
		}
		assignment.Y[i] = singles[i].Y
		assignment.ZTable[i] = singles[i].ZTable
		assignment.Rem[i] = singles[i].Rem
	}
	return assignment
}

// ComputeBatchLabelWitness fills a BatchLabelCircuit (label-only) assignment.
func ComputeBatchLabelWitness(wFloat []float64, bFloat float64, features [][]int, batchSize int) *circuit.BatchLabelCircuit {
	numFeatures := len(wFloat)
	singles := computeBatchSingles(wFloat, bFloat, features, batchSize)

	assignment := circuit.NewBatchLabelCircuit(batchSize, numFeatures)
	assignment.W = singles[0].W
	assignment.B = singles[0].B
	assignment.Commitment = singles[0].Commitment

	for i := 0; i < batchSize; i++ {
		for j := 0; j < numFeatures; j++ {
			assignment.X[i][j] = singles[i].X[j]
		}
		assignment.Label[i] = labelOf(singles[i])
		assignment.ZTable[i] = singles[i].ZTable
		assignment.Rem[i] = singles[i].Rem
	}
	return assignment
}

// labelOf returns the class bit (0/1) of a single witness.
func labelOf(c *circuit.LRCircuit) int {
	if c.Y.(*big.Int).Int64() >= 1<<(circuit.OutputPrecision-1) {
		return 1
	}
	return 0
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

	fmt.Printf("    Batches: %d (size=%d, workers=%d, keys=%d, mode=%s)\n", len(batches), batchSize, numWorkers, keyPoolSize, setup.Mode)
	fmt.Printf("    Memory: %.0f MB for key pool (%.1f MB × %d copies)\n", totalMemMB, pkSizeMB, keyPoolSize)

	// Create key pool - hybrid approach allows more workers than keys.
	// Workers wait for available keys, enabling memory-efficient parallelism.
	// A clone failure shrinks the pool instead of sharing one key across
	// workers (concurrent use of a shared key is not safe).
	keyPool := make(chan *keySet, keyPoolSize)
	pooled := 0
	for i := 0; i < keyPoolSize; i++ {
		keys, err := cloneKeys(setup.ProvingKey, setup.VerificationKey, setup.ConstraintSystem)
		if err != nil {
			fmt.Printf("    Warning: failed to clone keys %d, shrinking pool: %v\n", i, err)
			continue
		}
		keyPool <- keys
		pooled++
	}
	if pooled == 0 {
		// No clone succeeded — fall back to sequential (1 worker, original cs).
		fmt.Println("    Warning: key cloning failed entirely; running with a single shared key (1 concurrent prover)")
		keyPool <- &keySet{pk: setup.ProvingKey, vk: setup.VerificationKey, cs: setup.ConstraintSystem}
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
			singles := computeBatchSingles(wFloat, bFloat, batchFeatures, batchSize)

			// Build per-prediction results (prover-side view; in label mode
			// only the label is part of the public statement)
			for i, f := range batchFeatures {
				prob := GetProbability(singles[i])
				pred := "NORMAL"
				if labelOf(singles[i]) == 1 {
					pred = "OVERWEIGHT"
				}

				// Grab first two features for log formatting if they exist, otherwise pad 0
				h := 0
				w := 0
				if len(f) > 0 {
					h = f[0]
				}
				if len(f) > 1 {
					w = f[1]
				}

				result.Predictions = append(result.Predictions, &PredictionResult{
					Height:      h,
					Weight:      w,
					Probability: prob,
					Label:       labelOf(singles[i]),
					Prediction:  pred,
				})
			}

			// Assemble the mode-specific assignment
			var assignment frontend.Circuit
			switch setup.Mode {
			case ModeLabel:
				assignment = ComputeBatchLabelWitness(wFloat, bFloat, batchFeatures, batchSize)
			default:
				assignment = ComputeBatchWitness(wFloat, bFloat, batchFeatures, batchSize)
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

			// Prove using worker's private key copy and its own CS clone.
			proveStart := time.Now()
			proof, err := plonk.Prove(keys.cs, keys.pk, witness)
			result.ProveTime = time.Since(proveStart)

			if err != nil {
				result.Error = fmt.Errorf("batch %d prove failed: %w", idx, err)
				results[idx] = result
				return
			}

			var proofBuf bytes.Buffer
			if _, err := proof.WriteTo(&proofBuf); err != nil {
				result.Error = fmt.Errorf("batch %d proof serialization failed: %w", idx, err)
				results[idx] = result
				return
			}
			result.ProofBytes = proofBuf.Bytes()

			// Extract + serialize the public witness (for independent verification)
			publicWitness, err := witness.Public()
			if err != nil {
				result.Error = fmt.Errorf("batch %d public witness failed: %w", idx, err)
				results[idx] = result
				return
			}
			var pubBuf bytes.Buffer
			if _, err := publicWitness.WriteTo(&pubBuf); err != nil {
				result.Error = fmt.Errorf("batch %d public witness serialization failed: %w", idx, err)
				results[idx] = result
				return
			}
			result.PublicWitnessBytes = pubBuf.Bytes()

			// Verify using worker's private key copy
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
