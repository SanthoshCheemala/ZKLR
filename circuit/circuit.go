// Package circuit defines the ZK constraint system for logistic regression prediction.
//
// The circuit proves: "Given public input X and public output Y, the weights
// W, B committed to in the public Commitment satisfy Y = sigmoid(W*X + B)."
//
// Architecture:
//
//	┌──────────────────────────────────────────────┐
//	│            LR Prediction Circuit             │
//	│                                              │
//	│  Private: W (weights), B (bias)              │
//	│  Public:  X (features), Y (sigmoid out),     │
//	│           Commitment = MiMC(W..., B)         │
//	│                                              │
//	│  Constraint 1: Commitment = MiMC(W..., B)    │
//	│  Constraint 2: z = Σ W_i·X_i + B (+offset)   │
//	│  Constraint 3: truncate z to table precision │
//	│  Constraint 4: Y = SigmoidLookup(z)          │
//	└──────────────────────────────────────────────┘
package circuit

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
)

// ─── Scaling Constants ────────────────────────────────────────
//
// All values are pre-scaled by 2^32 (scalingFactor).
// The sigmoid lookup table uses 2^10 precision for input indexing
// and 2^16 precision for output values.

const Precision = 32      // W, B are scaled by 2^32
const InputPrecision = 10 // sigmoid table index precision
const OutputPrecision = 16 // sigmoid output precision

var ScalingFactor = new(big.Int).Lsh(big.NewInt(1), Precision) // 2^32

// ─── Circuit Definition ──────────────────────────────────────

// LRCircuit defines the constraint system for a single LR prediction.
//
// The prover must show that Y = sigmoid(W*X + B) where:
//   - W, B are secret model weights (scaled by 2^32)
//   - Commitment = MiMC(W..., B) binds the proof to one published model
//   - X is a public input (features, scaled integers)
//   - Y is the public sigmoid output (scaled by 2^16)
type LRCircuit struct {
	// Private inputs (model weights — kept secret)
	W []frontend.Variable `gnark:",secret"` // Weights
	B frontend.Variable   `gnark:",secret"`

	// Auxiliary private inputs for truncation
	ZTable frontend.Variable `gnark:",secret"` // z_shifted / 2^22
	Rem    frontend.Variable `gnark:",secret"` // remainder

	// Public inputs (visible to verifier)
	X          []frontend.Variable `gnark:",public"` // Features
	Y          frontend.Variable   `gnark:",public"` // sigmoid output (scaled by 2^16)
	Commitment frontend.Variable   `gnark:",public"` // MiMC(W..., B)
}

// NewLRCircuit creates an empty circuit assignment for a given number of features.
// This is required so gnark knows exactly how to size the slices during constraint generation.
func NewLRCircuit(numFeatures int) *LRCircuit {
	return &LRCircuit{
		W: make([]frontend.Variable, numFeatures),
		X: make([]frontend.Variable, numFeatures),
	}
}

// Define builds the constraint system.
//
// This is called by gnark's compiler to generate the SCS (Sparse Constraint System).
func (c *LRCircuit) Define(api frontend.API) error {
	if err := assertCommitment(api, c.W, c.B, c.Commitment); err != nil {
		return err
	}
	table := newSigmoidTable(api)
	y := definePrediction(api, table, c.W, c.B, c.X, c.ZTable, c.Rem)
	api.AssertIsEqual(c.Y, y)
	return nil
}
