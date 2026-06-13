// gadget.go — Shared per-prediction constraint logic.
//
// Both the single circuit (LRCircuit) and the batch circuits build the same
// per-sample constraints; keeping them in one gadget guarantees the batch
// circuit is constraint-for-constraint identical to the audited single case.
package circuit

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
)

// newSigmoidTable builds the in-circuit sigmoid lookup table (built once per
// circuit; shared by every prediction in a batch).
func newSigmoidTable(api frontend.API) *logderivlookup.Table {
	table := logderivlookup.New(api)
	for _, entry := range BuildSigmoidTable() {
		table.Insert(entry)
	}
	return table
}

// definePrediction adds the constraints for one prediction and returns the
// looked-up sigmoid output y (scaled by 2^16):
//
//  1. Range check features:        X_j ∈ [0, 2^12)
//  2. Linear layer:                z_shifted = Σ W_j·X_j + B + ModelOffset·2^32
//  3. Truncation to table scale:   z_shifted = zTable·2^22 + rem,
//     with rem ∈ [0, 2^22) and zTable ∈ [0, 2^24) — the unique decomposition
//     given that z_shifted < 2^46
//  4. Clamp zTable to the table window [(ModelOffset−SigmoidOffset)·2^10,
//     (ModelOffset+SigmoidOffset)·2^10] (sigmoid is saturated outside)
//  5. Lookup:                      y = sigmoidTable[zTable − windowLow]
func definePrediction(
	api frontend.API,
	table *logderivlookup.Table,
	w []frontend.Variable, b frontend.Variable,
	x []frontend.Variable,
	zTable, rem frontend.Variable,
) frontend.Variable {
	// Step 1: range check X features
	for j := 0; j < len(x); j++ {
		api.ToBinary(x[j], 12)
	}

	// Step 2: z_shifted = sum(W_j * X_j) + B + (ModelOffset * 2^32)
	// X is unscaled, so W*X is naturally at the 2^32 scale of W.
	wx := frontend.Variable(0)
	for j := 0; j < len(x); j++ {
		wx = api.Add(wx, api.Mul(w[j], x[j]))
	}
	zLinear := api.Add(wx, b)
	offsetScaled := new(big.Int).Lsh(big.NewInt(ModelOffset), Precision)
	zShifted := api.Add(zLinear, offsetScaled)

	// Step 3: truncation z_shifted = zTable * 2^22 + rem
	api.ToBinary(rem, Precision-InputPrecision)
	shiftFactor := new(big.Int).Lsh(big.NewInt(1), Precision-InputPrecision)
	zReconstructed := api.Add(api.Mul(zTable, shiftFactor), rem)
	api.AssertIsEqual(zShifted, zReconstructed)

	// Bound zTable so api.Cmp below is sound (24 bits ≫ window upper bound).
	api.ToBinary(zTable, 24)

	// Step 4: clamp to the valid table window
	lowerBound := (ModelOffset - SigmoidOffset) * (1 << InputPrecision)
	upperBound := (ModelOffset + SigmoidOffset) * (1 << InputPrecision)

	cmpLower := api.Cmp(zTable, lowerBound)
	isBelow := api.IsZero(api.Add(cmpLower, 1))
	cmpUpper := api.Cmp(zTable, upperBound)
	isAbove := api.IsZero(api.Sub(cmpUpper, 1))

	val1 := api.Select(isBelow, lowerBound, zTable)
	zClamped := api.Select(isAbove, upperBound, val1)
	zIndex := api.Sub(zClamped, lowerBound)

	// Step 5: sigmoid lookup
	return table.Lookup(zIndex)[0]
}

// labelBit returns the class bit of a sigmoid output y (scaled by 2^16):
// 1 iff y >= 2^15, i.e. probability >= 0.5. Exposing only this bit (instead
// of the full-precision probability) prevents model-extraction attacks that
// solve logit(Y) = W·X + B from observed predictions.
func labelBit(api frontend.API, y frontend.Variable) frontend.Variable {
	bits := api.ToBinary(y, OutputPrecision)
	return bits[OutputPrecision-1]
}
