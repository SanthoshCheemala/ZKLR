// sigmoid.go — Pre-computed sigmoid lookup table for the ZK circuit.
//
// The sigmoid function σ(z) = 1 / (1 + e^(-z)) cannot be computed
// directly inside a ZK circuit (no exponentiation, no floating-point).
//
// Solution: Pre-compute a table of sigmoid values at compile time.
// At proving time, the circuit looks up the result instead of computing it.
//
// Table structure:
//
//	Index i → sigmoid(i / 2^10 - SigmoidOffset), scaled by 2^16
//
//	Example: i = 11264 → z = 11264/1024 - 10 = 1.0 → sigmoid(1.0) = 0.7311 → stored as 47910
//
// The table covers z ∈ [-SigmoidOffset, +SigmoidOffset] = [-10, +10]
// (20·1024 + 1 = 20481 entries). Outside this range sigmoid is saturated
// (< 5e-5 from 0 or 1); the circuit clamps the index to the table bounds.
package circuit

import (
	"math"
	"math/big"
)

// ModelOffset is added to Z inside the circuit to make it strictly positive.
// This allows us to avoid negative numbers which wrap around in finite fields.
const ModelOffset = 1000

// SigmoidOffset is used for table bounds [-10, +10]
const SigmoidOffset = 10

// SigmoidRange is the total range of Z covered by the table (from -10 to +10).
const SigmoidRange = 20

// BuildSigmoidTable returns a slice of *big.Int values representing
// sigmoid(z) for z ∈ [-10, +10] at InputPrecision (2^10) resolution.
//
// Index i represents z = (i / 2^10) - 10
// Total entries = 20 * 1024 + 1 = 20481
func BuildSigmoidTable() []*big.Int {
	tableSize := SigmoidRange*(1<<InputPrecision) + 1 // 20*1024 + 1 = 20481 entries
	table := make([]*big.Int, tableSize)

	for i := 0; i < tableSize; i++ {
		// Index i represents z_shifted. Actual z is z_shifted - 10.0
		z := float64(i)/float64(1<<InputPrecision) - float64(SigmoidOffset)

		// Compute sigmoid(z)
		y := 1.0 / (1.0 + math.Exp(-z))

		// Scale to integer: y × 2^16, clamped to [0, 2^16 - 1]
		yScaled := int64(y * float64(1<<OutputPrecision))
		maxVal := int64((1 << OutputPrecision) - 1)
		if yScaled > maxVal {
			yScaled = maxVal
		}

		table[i] = big.NewInt(yScaled)
	}

	return table
}

// SigmoidFloat computes sigmoid(z) in float64.
func SigmoidFloat(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
