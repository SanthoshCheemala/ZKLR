// Package circuit defines the ZK constraint system for logistic regression prediction.
//
// The circuit proves: "Given public input X (marks) and public output Y (prediction),
// there exist secret weights W, B such that Y = sigmoid(W*X + B)."
//
// Architecture:
//
//	┌─────────────────────────────────────────┐
//	│           LR Prediction Circuit         │
//	│                                         │
//	│  Private: W (weight), B (bias)          │
//	│  Public:  X (marks),  Y (sigmoid out)   │
//	│                                         │
//	│  Constraint 1: z_raw = W * X            │
//	│  Constraint 2: z_lin = z_raw / 2^32 + B │
//	│  Constraint 3: z = z_lin / 2^22         │
//	│  Constraint 4: Y = SigmoidLookup(z)     │
//	└─────────────────────────────────────────┘
package circuit

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/lookup/logderivlookup"
)

// ─── Scaling Constants ────────────────────────────────────────
//
// All values are pre-scaled by 2^32 (scalingFactor).
// The sigmoid lookup table uses 2^10 precision for input indexing
// and 2^16 precision for output values.

const Precision = 32                                                     // W, B are scaled by 2^32
const InputPrecision = 10                                                // sigmoid table index precision
const OutputPrecision = 16                                               // sigmoid output precision

var ScalingFactor = new(big.Int).Lsh(big.NewInt(1), Precision)           // 2^32

// ─── Circuit Definition ──────────────────────────────────────

// LRCircuit defines the constraint system for a single LR prediction.
//
// The prover must show that Y = sigmoid(W*X + B) where:
//   - W, B are secret model weights (scaled by 2^32)
//   - X is a public input (marks, scaled by 2^32)
//   - Y is the public sigmoid output (scaled by 2^16)
type LRCircuit struct {
	// Private inputs (model weights — kept secret)
	W [2]frontend.Variable `gnark:",secret"` // Weights: W1 (height), W2 (weight)
	B frontend.Variable    `gnark:",secret"`

	// Auxiliary private inputs for truncation
	ZTable frontend.Variable `gnark:",secret"` // z_shifted / 2^22
	Rem    frontend.Variable `gnark:",secret"` // remainder

	// Public inputs (visible to verifier)
	X [2]frontend.Variable `gnark:",public"` // Features: X1 (height), X2 (weight)
	Y frontend.Variable    `gnark:",public"` // sigmoid output (scaled by 2^16)
}

// Define builds the constraint system.
//
// This is called by gnark's compiler to generate the SCS (Sparse Constraint System).
// Every api.Mul / api.Add / api.Div call adds constraints that the prover must satisfy.
func (c *LRCircuit) Define(api frontend.API) error {
	// ───────────────────────────────────────────────────
	// Step 1: Range check X features
	// ───────────────────────────────────────────────────
	// X1 (Height): max ~250cm -> fits in 8 bits
	api.ToBinary(c.X[0], 8)
	// X2 (Weight*10): max ~2000 -> fits in 12 bits
	api.ToBinary(c.X[1], 12)

	// ───────────────────────────────────────────────────
	// Step 2: Compute z_shifted = W1*X1 + W2*X2 + B + (10 * 2^32)
	// ───────────────────────────────────────────────────
	// Since X is unscaled, W*X is naturally scaled by 2^32. No division needed!
	w1x1 := api.Mul(c.W[0], c.X[0])
	w2x2 := api.Mul(c.W[1], c.X[1])
	wx := api.Add(w1x1, w2x2)
	z_linear := api.Add(wx, c.B)
	
	// Add offset to make strictly positive. ModelOffset * 2^32:
	offsetScaled := new(big.Int).Lsh(big.NewInt(ModelOffset), Precision)
	z_shifted := api.Add(z_linear, offsetScaled)

	// ───────────────────────────────────────────────────
	// Step 3: Truncate z_shifted from 2^32 to 2^10 for lookup
	// ───────────────────────────────────────────────────
	// z_shifted = ZTable * 2^22 + Rem
	// Enforce 0 <= Rem < 2^22 (fits in 22 bits)
	api.ToBinary(c.Rem, 22)

	shiftFactor := new(big.Int).Lsh(big.NewInt(1), 22) // 2^22
	z_reconstructed := api.Add(api.Mul(c.ZTable, shiftFactor), c.Rem)
	api.AssertIsEqual(z_shifted, z_reconstructed)

	// Range check ZTable to prevent overflow in Cmp (e.g. 24 bits maximum)
	// 1000 * 2^10 = 1024000 which fits in 20 bits. Bound to 24 bits.
	api.ToBinary(c.ZTable, 24)

	// ───────────────────────────────────────────────────
	// Step 4: Clamp ZTable to valid table range [LowerBound, UpperBound]
	// ───────────────────────────────────────────────────
	LowerBound := (ModelOffset - SigmoidOffset) * (1 << InputPrecision)
	UpperBound := (ModelOffset + SigmoidOffset) * (1 << InputPrecision)

	cmpLower := api.Cmp(c.ZTable, LowerBound)
	isBelow := api.IsZero(api.Add(cmpLower, 1))

	cmpUpper := api.Cmp(c.ZTable, UpperBound)
	isAbove := api.IsZero(api.Sub(cmpUpper, 1))

	val1 := api.Select(isBelow, LowerBound, c.ZTable)
	z_clamped := api.Select(isAbove, UpperBound, val1)

	z_index := api.Sub(z_clamped, LowerBound)

	// ───────────────────────────────────────────────────
	// Step 5: Sigmoid lookup — Y = sigmoid(z)
	// ───────────────────────────────────────────────────
	y_pos := SigmoidLookup(api, z_index)

	// Constrain: public Y must match the lookup result
	api.AssertIsEqual(c.Y, y_pos)

	return nil
}

// SigmoidLookup builds the sigmoid lookup table inside the circuit
// and returns sigmoid(z) for the given z (at 2^10 scale).
//
// The table covers exactly the shifted Z range with 20481 entries.
func SigmoidLookup(api frontend.API, z frontend.Variable) frontend.Variable {
	table := logderivlookup.New(api)

	// Populate lookup table with pre-computed sigmoid values
	entries := BuildSigmoidTable()
	for _, entry := range entries {
		table.Insert(entry)
	}

	// Look up sigmoid(z) — returns a slice, we take the first result
	results := table.Lookup(z)
	return results[0]
}
