// batch.go — Batch circuits: prove N predictions in a single PLONK proof.
//
// The key insight: the sigmoid lookup table (20,481 entries ≈ 105K constraints)
// is the dominant cost. By batching N predictions into one circuit, we share
// the table cost across all N predictions.
//
// Cost breakdown:
//   - 1 prediction:   ~108K constraints (105K table + 3K logic)
//   - N predictions:  ~105K + N*3K constraints (table shared!)
//
// Two variants share the same prediction gadget:
//
//   - BatchCircuit:      exposes the full-precision probability Y per sample.
//     WARNING: publishing exact probabilities allows model extraction
//     (solve logit(Y) = W·X + B from ~d+1 observations). Use only when the
//     verifier is trusted with the probabilities.
//   - BatchLabelCircuit: exposes only the class bit per sample (Y stays
//     internal to the circuit). Default for privacy-preserving deployments.
package circuit

import (
	"github.com/consensys/gnark/frontend"
)

// BatchCircuit proves N logistic regression predictions in a single proof,
// exposing full-precision probabilities as public outputs.
// All predictions share the same sigmoid lookup table and the same W, B.
type BatchCircuit struct {
	// Batch size (set at compile time, not a variable)
	BatchSize int `gnark:"-"`

	// Private inputs (model weights — same for all predictions)
	W []frontend.Variable `gnark:",secret"`
	B frontend.Variable   `gnark:",secret"`

	// Per-prediction inputs
	X      [][]frontend.Variable `gnark:",public"` // features for each sample
	Y      []frontend.Variable   `gnark:",public"` // sigmoid output for each
	ZTable []frontend.Variable   `gnark:",secret"` // lookup index for each
	Rem    []frontend.Variable   `gnark:",secret"` // remainder for each

	// Commitment = MiMC(W..., B) binds all proofs to one published model.
	Commitment frontend.Variable `gnark:",public"`
}

// NewBatchCircuit creates a BatchCircuit with allocated slices.
func NewBatchCircuit(batchSize int, numFeatures int) *BatchCircuit {
	xSlices := make([][]frontend.Variable, batchSize)
	for i := range xSlices {
		xSlices[i] = make([]frontend.Variable, numFeatures)
	}
	return &BatchCircuit{
		BatchSize: batchSize,
		W:         make([]frontend.Variable, numFeatures),
		X:         xSlices,
		Y:         make([]frontend.Variable, batchSize),
		ZTable:    make([]frontend.Variable, batchSize),
		Rem:       make([]frontend.Variable, batchSize),
	}
}

// Define builds the batch constraint system.
func (c *BatchCircuit) Define(api frontend.API) error {
	if err := assertCommitment(api, c.W, c.B, c.Commitment); err != nil {
		return err
	}
	table := newSigmoidTable(api) // built ONCE, shared by all predictions
	for i := 0; i < c.BatchSize; i++ {
		y := definePrediction(api, table, c.W, c.B, c.X[i], c.ZTable[i], c.Rem[i])
		api.AssertIsEqual(c.Y[i], y)
	}
	return nil
}

// BatchLabelCircuit proves N predictions exposing only the class bit
// (1 iff probability >= 0.5) per sample. The full-precision sigmoid output
// never leaves the circuit, so observers cannot reconstruct the weights by
// solving logit(Y) = W·X + B.
type BatchLabelCircuit struct {
	BatchSize int `gnark:"-"`

	W []frontend.Variable `gnark:",secret"`
	B frontend.Variable   `gnark:",secret"`

	X      [][]frontend.Variable `gnark:",public"` // features for each sample
	Label  []frontend.Variable   `gnark:",public"` // class bit for each (0/1)
	ZTable []frontend.Variable   `gnark:",secret"` // lookup index for each
	Rem    []frontend.Variable   `gnark:",secret"` // remainder for each

	Commitment frontend.Variable `gnark:",public"` // MiMC(W..., B)
}

// NewBatchLabelCircuit creates a BatchLabelCircuit with allocated slices.
func NewBatchLabelCircuit(batchSize int, numFeatures int) *BatchLabelCircuit {
	xSlices := make([][]frontend.Variable, batchSize)
	for i := range xSlices {
		xSlices[i] = make([]frontend.Variable, numFeatures)
	}
	return &BatchLabelCircuit{
		BatchSize: batchSize,
		W:         make([]frontend.Variable, numFeatures),
		X:         xSlices,
		Label:     make([]frontend.Variable, batchSize),
		ZTable:    make([]frontend.Variable, batchSize),
		Rem:       make([]frontend.Variable, batchSize),
	}
}

// Define builds the label-only batch constraint system.
func (c *BatchLabelCircuit) Define(api frontend.API) error {
	if err := assertCommitment(api, c.W, c.B, c.Commitment); err != nil {
		return err
	}
	table := newSigmoidTable(api)
	for i := 0; i < c.BatchSize; i++ {
		y := definePrediction(api, table, c.W, c.B, c.X[i], c.ZTable[i], c.Rem[i])
		api.AssertIsEqual(c.Label[i], labelBit(api, y))
	}
	return nil
}
