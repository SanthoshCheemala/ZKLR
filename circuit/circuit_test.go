// circuit_test.go — Unit tests for the LR prediction circuit.
//
// Tests verify that the constraint system accepts valid witnesses
// and rejects invalid ones. This validates circuit correctness
// independently of the main application.
package circuit

import (
	"math"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test"
)

// ─── Helper: compute witness values outside the circuit ───────

// computeWitness calculates z and y for given w, b, marks.
// All inputs/outputs are pre-scaled integers matching the circuit's expectations.
func computeWitness(w1Float, w2Float, bFloat float64, height, weight int) (w_bi [2]*big.Int, b_bi *big.Int, x_bi [2]*big.Int, z_table, rem, y_bi *big.Int) {
	scale := new(big.Int).Set(ScalingFactor) // 2^32

	// Scale w, b to 2^32. x is unscaled.
	w_bi[0] = new(big.Int).SetInt64(int64(w1Float * float64(scale.Int64())))
	w_bi[1] = new(big.Int).SetInt64(int64(w2Float * float64(scale.Int64())))
	b_bi = new(big.Int).SetInt64(int64(bFloat * float64(scale.Int64())))
	x_bi[0] = big.NewInt(int64(height))
	x_bi[1] = big.NewInt(int64(weight))

	// z_linear = W1*H + W2*W + B  (at 2^32 scale)
	w1x1 := new(big.Int).Mul(w_bi[0], x_bi[0])
	w2x2 := new(big.Int).Mul(w_bi[1], x_bi[1])
	z_linear := new(big.Int).Add(w1x1, w2x2)
	z_linear.Add(z_linear, b_bi)

	// z_shifted = z_linear + (ModelOffset * 2^32)
	offsetScaled := new(big.Int).Lsh(big.NewInt(ModelOffset), Precision)
	z_shifted := new(big.Int).Add(z_linear, offsetScaled)

	// Shift from 2^32 to 2^10 -> shift by 22 bits
	shiftFactor := new(big.Int).Lsh(big.NewInt(1), Precision-InputPrecision)
	
	z_table = new(big.Int).Div(z_shifted, shiftFactor)
	rem = new(big.Int).Rem(z_shifted, shiftFactor)

	// Clamp z_table 
	LowerBound := int64((ModelOffset - SigmoidOffset) * (1 << InputPrecision))
	UpperBound := int64((ModelOffset + SigmoidOffset) * (1 << InputPrecision))

	z_table_clamped := new(big.Int).Set(z_table)
	if z_table_clamped.Int64() < LowerBound {
		z_table_clamped = big.NewInt(LowerBound)
	} else if z_table_clamped.Int64() > UpperBound {
		z_table_clamped = big.NewInt(UpperBound)
	}

	z_index := z_table_clamped.Int64() - LowerBound

	// y = sigmoid(z_f) scaled by 2^16
	z_f := float64(z_index) / float64(1<<InputPrecision) - float64(SigmoidOffset)
	y_float := SigmoidFloat(z_f)
	y_bi = new(big.Int).SetInt64(int64(y_float * float64(1<<OutputPrecision)))

	// Clamp output
	maxOut := big.NewInt((1 << OutputPrecision) - 1)
	if y_bi.Cmp(maxOut) > 0 {
		y_bi = maxOut
	}

	return
}

// ─── Test: Circuit Compiles ──────────────────────────────────

func TestCircuitCompiles(t *testing.T) {
	var c LRCircuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &c)
	if err != nil {
		t.Fatalf("Circuit compilation failed: %v", err)
	}
	t.Logf("Circuit compiled: %d constraints", ccs.GetNbConstraints())
}

// ─── Test: Valid Witness (Passing Student) ────────────────────

func TestValidWitnessPass(t *testing.T) {
	// h=160, w×10=850 (85.0kg, BMI 33.2) → OVERWEIGHT
	w, b, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 160, 850)

	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: b, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Valid witness (overweight) rejected: %v", err)
	}
	t.Logf("h=160, w×10=850: z_table=%v, y=%v (overweight)", zt, y)
}

// ─── Test: Valid Witness (Failing Student) ────────────────────

func TestValidWitnessFail(t *testing.T) {
	// h=180, w×10=600 (60.0kg, BMI 18.5) → NORMAL
	w, b, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 180, 600)

	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: b, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Valid witness (normal) rejected: %v", err)
	}
	t.Logf("h=180, w×10=600: z_table=%v, y=%v (normal)", zt, y)
}

// ─── Test: Valid Witness (Boundary Student) ──────────────────

func TestValidWitnessBoundary(t *testing.T) {
	// h=170, w×10=725 (72.5kg, BMI ~25.1) → borderline
	w, b, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 725)

	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: b, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Valid witness (boundary) rejected: %v", err)
	}
	t.Logf("boundary: z_table=%v, y=%v (boundary)", zt, y)
}

// ─── Test: Wrong Y Rejects ───────────────────────────────────

func TestInvalidY(t *testing.T) {
	w, b, x, zt, rem, _ := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)

	// Tamper Y with a wrong value
	wrongY := big.NewInt(12345)

	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: b, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: wrongY}
	err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("Circuit should reject wrong Y but didn't")
	}
	t.Logf("Correctly rejected wrong Y: %v", err)
}

// ─── Test: Wrong Z Rejects ───────────────────────────────────

func TestInvalidZ(t *testing.T) {
	w, b, x, _, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)

	// Tamper ZTable
	wrongZ := big.NewInt(999)

	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: b, X: [2]frontend.Variable{x[0], x[1]}, ZTable: wrongZ, Rem: rem, Y: y}
	err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("Circuit should reject wrong Z but didn't")
	}
	t.Logf("Correctly rejected wrong Z: %v", err)
}

// ─── Test: Sigmoid Table Correctness ─────────────────────────

func TestSigmoidTable(t *testing.T) {
	table := BuildSigmoidTable()

	// Check table size
	expectedSize := SigmoidRange*(1<<InputPrecision) + 1 // 20481
	if len(table) != expectedSize {
		t.Fatalf("Expected %d entries, got %d", expectedSize, len(table))
	}

	offset := SigmoidOffset * (1 << InputPrecision) // 10 * 1024 = 10240

	// Spot-check values (offset = 10 * 1024 = 10240)
	checks := []struct {
		index    int
		expected float64 // sigmoid value (unscaled)
		name     string
	}{
		{offset, 0.5, "sigmoid(0)"},
		{offset + 1024, 1.0 / (1.0 + math.Exp(-1.0)), "sigmoid(1)"},
		{offset + 2048, 1.0 / (1.0 + math.Exp(-2.0)), "sigmoid(2)"},
		{offset + 4096, 1.0 / (1.0 + math.Exp(-4.0)), "sigmoid(4)"},
		{offset - 4096, 1.0 / (1.0 + math.Exp(4.0)), "sigmoid(-4)"},
		{0, SigmoidFloat(-float64(SigmoidOffset)), "sigmoid(-10)"},
	}

	for _, c := range checks {
		actual := float64(table[c.index].Int64()) / float64(1<<OutputPrecision)
		diff := math.Abs(actual - c.expected)
		if diff > 0.001 { // allow 0.1% error from integer rounding
			t.Errorf("%s: expected %.4f, got %.4f (diff=%.6f)", c.name, c.expected, actual, diff)
		} else {
			t.Logf("%s: %.4f ✓ (diff=%.6f)", c.name, actual, diff)
		}
	}
}

// ─── Test: Multiple Features Range ──────────────────────────

func TestMultipleFeatures(t *testing.T) {
	dataToTest := [][2]int{
		{150, 500}, {160, 600}, {170, 750}, {180, 850}, {190, 1000},
	}

	for _, features := range dataToTest {
		h, w := features[0], features[1]
		w_bi, b_bi, x_bi, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, h, w)
		assignment := &LRCircuit{W: [2]frontend.Variable{w_bi[0], w_bi[1]}, B: b_bi, X: [2]frontend.Variable{x_bi[0], x_bi[1]}, ZTable: zt, Rem: rem, Y: y}
		err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())

		yFloat := float64(y.Int64()) / float64(1<<OutputPrecision)
		prediction := "NORMAL"
		if yFloat >= 0.5 {
			prediction = "OVERWEIGHT"
		}

		if err != nil {
			t.Errorf("h=%d, w=%d: REJECTED (%v)", h, w, err)
		} else {
			t.Logf("h=%3d, w=%3d: z_table=%-6v y=%.4f → %s ✓", h, w, zt, yFloat, prediction)
		}
	}
}
