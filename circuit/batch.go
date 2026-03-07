// batch.go — Batch circuit: proves N predictions in a single PLONK proof.
//
// The key insight: the sigmoid lookup table (20,481 entries = ~105K constraints)
// is the dominant cost. By batching N predictions into one circuit, we share
// the table cost across all N predictions.
//
// Cost breakdown:
//   - 1 prediction:   ~108K constraints (105K table + 3K logic)
//   - N predictions:  ~105K + N*3K constraints (table shared!)
//
// This means batch=20 only adds ~60K constraints vs 20 × 108K = 2.16M.
package circuit

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
)

// BatchCircuit proves N logistic regression predictions in a single proof.
// All predictions share the same sigmoid lookup table.
type BatchCircuit struct {
	// Batch size (set at compile time, not a variable)
	BatchSize int `gnark:"-"`

	// Private inputs (model weights — same for all predictions)
	W [2]frontend.Variable `gnark:",secret"`
	B frontend.Variable    `gnark:",secret"`

	// Per-prediction inputs
	X      [][2]frontend.Variable `gnark:",public"`  // features for each student
	Y      []frontend.Variable `gnark:",public"`  // sigmoid output for each
	ZTable []frontend.Variable `gnark:",secret"`  // lookup index for each
	Rem    []frontend.Variable `gnark:",secret"`  // remainder for each
}

// NewBatchCircuit creates a BatchCircuit with allocated slices.
func NewBatchCircuit(batchSize int) *BatchCircuit {
	return &BatchCircuit{
		BatchSize: batchSize,
		X:         make([][2]frontend.Variable, batchSize),
		Y:         make([]frontend.Variable, batchSize),
		ZTable:    make([]frontend.Variable, batchSize),
		Rem:       make([]frontend.Variable, batchSize),
	}
}

// Define builds the batch constraint system.
func (c *BatchCircuit) Define(api frontend.API) error {
	// ─── Build sigmoid table ONCE (shared across all predictions) ───
	table := logderivlookup.New(api)
	entries := BuildSigmoidTable()
	for _, entry := range entries {
		table.Insert(entry)
	}

	offsetScaled := new(big.Int).Lsh(big.NewInt(ModelOffset), Precision)
	shiftFactor := new(big.Int).Lsh(big.NewInt(1), Precision-InputPrecision)

	// ─── Apply constraints for each prediction ───
	for i := 0; i < c.BatchSize; i++ {
		// Range check X features
		api.ToBinary(c.X[i][0], 8)
		api.ToBinary(c.X[i][1], 12)

		// z_shifted = W1*X1 + W2*X2 + B + offset
		w1x1 := api.Mul(c.W[0], c.X[i][0])
		w2x2 := api.Mul(c.W[1], c.X[i][1])
		wx := api.Add(w1x1, w2x2)
		zLinear := api.Add(wx, c.B)
		zShifted := api.Add(zLinear, offsetScaled)

		// Truncation: z_shifted = ZTable * 2^22 + Rem
		api.ToBinary(c.Rem[i], 22)
		zReconstructed := api.Add(api.Mul(c.ZTable[i], shiftFactor), c.Rem[i])
		api.AssertIsEqual(zShifted, zReconstructed)

		api.ToBinary(c.ZTable[i], 24)

		LowerBound := (ModelOffset - SigmoidOffset) * (1 << InputPrecision)
		UpperBound := (ModelOffset + SigmoidOffset) * (1 << InputPrecision)

		cmpLower := api.Cmp(c.ZTable[i], LowerBound)
		isBelow := api.IsZero(api.Add(cmpLower, 1))

		cmpUpper := api.Cmp(c.ZTable[i], UpperBound)
		isAbove := api.IsZero(api.Sub(cmpUpper, 1))

		val1 := api.Select(isBelow, LowerBound, c.ZTable[i])
		zClamped := api.Select(isAbove, UpperBound, val1)

		zIndex := api.Sub(zClamped, LowerBound)
		results := table.Lookup(zIndex)
		api.AssertIsEqual(c.Y[i], results[0])
	}

	return nil
}
