// multiclass.go — One-vs-rest multi-class batch circuit (Phase 4).
//
// Generalizes the shared-lookup amortization to a second dimension: C binary
// classifiers (one per class) and B samples share ONE sigmoid table, so
//
//	constraints ≈ table + (B·C)·logic
//
// instead of B·C·(table + logic). Each public Label[i][c] is the one-vs-rest
// decision bit of classifier c on sample i (probability >= 0.5). The argmax /
// tie-breaking policy is left to the (public) consumer of the bits — every
// bit is individually proven, so any deterministic policy over them inherits
// the proof's integrity.
package circuit

import (
	"github.com/consensys/gnark/frontend"
)

// OneVsRestBatchCircuit proves B×C one-vs-rest predictions in a single proof.
type OneVsRestBatchCircuit struct {
	BatchSize  int `gnark:"-"`
	NumClasses int `gnark:"-"`

	// Private model: one weight row + bias per class
	W [][]frontend.Variable `gnark:",secret"` // [class][feature]
	B []frontend.Variable   `gnark:",secret"` // [class]

	// Per-sample public features, per-sample-per-class public decision bits
	X     [][]frontend.Variable `gnark:",public"` // [sample][feature]
	Label [][]frontend.Variable `gnark:",public"` // [sample][class]

	// Truncation witnesses per (sample, class)
	ZTable [][]frontend.Variable `gnark:",secret"` // [sample][class]
	Rem    [][]frontend.Variable `gnark:",secret"` // [sample][class]

	// Commitment = MiMC(W[0]..., W[C-1]..., B[0..C-1]) binds all classifiers.
	Commitment frontend.Variable `gnark:",public"`
}

// NewOneVsRestBatchCircuit allocates a circuit shape for compilation.
func NewOneVsRestBatchCircuit(batchSize, numClasses, numFeatures int) *OneVsRestBatchCircuit {
	w := make([][]frontend.Variable, numClasses)
	for c := range w {
		w[c] = make([]frontend.Variable, numFeatures)
	}
	x := make([][]frontend.Variable, batchSize)
	label := make([][]frontend.Variable, batchSize)
	zt := make([][]frontend.Variable, batchSize)
	rem := make([][]frontend.Variable, batchSize)
	for i := range x {
		x[i] = make([]frontend.Variable, numFeatures)
		label[i] = make([]frontend.Variable, numClasses)
		zt[i] = make([]frontend.Variable, numClasses)
		rem[i] = make([]frontend.Variable, numClasses)
	}
	return &OneVsRestBatchCircuit{
		BatchSize:  batchSize,
		NumClasses: numClasses,
		W:          w,
		B:          make([]frontend.Variable, numClasses),
		X:          x,
		Label:      label,
		ZTable:     zt,
		Rem:        rem,
	}
}

// Define builds the one-vs-rest constraint system.
func (c *OneVsRestBatchCircuit) Define(api frontend.API) error {
	if err := assertCommitmentMulti(api, c.W, c.B, c.Commitment); err != nil {
		return err
	}

	table := newSigmoidTable(api) // ONE table for all B×C evaluations

	for i := 0; i < c.BatchSize; i++ {
		for cls := 0; cls < c.NumClasses; cls++ {
			y := definePrediction(api, table, c.W[cls], c.B[cls], c.X[i], c.ZTable[i][cls], c.Rem[i][cls])
			api.AssertIsEqual(c.Label[i][cls], labelBit(api, y))
		}
	}
	return nil
}

