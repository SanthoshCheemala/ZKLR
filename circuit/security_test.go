// security_test.go — Negative tests: the tamper matrix.
//
// Each test feeds the circuit a witness with exactly one component tampered
// and asserts the constraint system rejects it. Together with the positive
// tests these form the empirical soundness evidence referenced in
// docs/correctness.md.
package circuit

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

const (
	secW1 = -3.3144933046
	secW2 = 0.3877500778
	secB  = 281.2861173099
)

// ─── Commitment binding ──────────────────────────────────────

// TestWrongCommitmentRejected: a proof must not verify against a commitment
// the weights don't match (an arbitrary tampered value).
func TestWrongCommitmentRejected(t *testing.T) {
	w, b, x, zt, rem, y := computeWitness(secW1, secW2, secB, 170, 750)
	assignment := newAssignment(w, b, x, zt, rem, y)
	assignment.Commitment = big.NewInt(123456789)

	err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("circuit accepted a wrong model commitment")
	}
	t.Logf("Correctly rejected tampered commitment: %v", err)
}

// TestModelSwapRejected: the model-binding property. A prover who computed a
// correct prediction with weights A cannot present it under the published
// commitment of weights B (the "swap the model after publishing" attack).
func TestModelSwapRejected(t *testing.T) {
	// Witness computed honestly with model A
	w, b, x, zt, rem, y := computeWitness(secW1, secW2, secB, 170, 750)
	assignment := newAssignment(w, b, x, zt, rem, y)

	// ...but presented under the commitment of a different model B
	wB, bB, _, _, _, _ := computeWitness(-1.5, 0.9, 100.0, 170, 750)
	assignment.Commitment = ComputeCommitment([]*big.Int{wB[0], wB[1]}, bB)

	err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("circuit accepted weights A under model B's commitment")
	}
	t.Logf("Correctly rejected model swap: %v", err)
}

// ─── Input range check ───────────────────────────────────────

// TestXBoundary12Bit: features at the 12-bit boundary. 4095 (max) must be
// accepted, 4096 must violate the ToBinary range check.
//
// Uses small weights so z stays inside the circuit's completeness domain
// (|W·X + B| < ModelOffset); the point here is the X range check alone.
func TestXBoundary12Bit(t *testing.T) {
	// x1 = 4095: maximum representable feature value
	w, b, x, zt, rem, y := computeWitness(0.001, 0, 0.5, 4095, 0)
	assignment := newAssignment(w, b, x, zt, rem, y)
	if err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField()); err != nil {
		t.Fatalf("x=4095 (12-bit max) should be accepted: %v", err)
	}
	t.Log("x=4095 accepted ✓")

	// x1 = 4096: out of range, must be rejected
	w, b, x, zt, rem, y = computeWitness(0.001, 0, 0.5, 4096, 0)
	assignment = newAssignment(w, b, x, zt, rem, y)
	if err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("x=4096 (13 bits) should violate the 12-bit range check")
	}
	t.Log("x=4096 rejected ✓")
}

// ─── Truncation witness ──────────────────────────────────────

// TestWrongRemRejected: tampering the truncation remainder must break the
// decomposition equality (or its range check).
func TestWrongRemRejected(t *testing.T) {
	w, b, x, zt, rem, y := computeWitness(secW1, secW2, secB, 170, 750)

	wrongRem := new(big.Int).Add(rem, big.NewInt(1))
	assignment := newAssignment(w, b, x, zt, wrongRem, y)
	if err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("circuit accepted a tampered remainder")
	}

	// Rem beyond 22 bits must violate the range check even if the
	// decomposition is rebalanced via ZTable.
	bigRem := new(big.Int).Lsh(big.NewInt(1), 22) // 2^22, one past the max
	shiftedZt := new(big.Int).Sub(zt, big.NewInt(1))
	rebalanced := newAssignment(w, b, x, shiftedZt, new(big.Int).Add(bigRem, rem), y)
	if err := test.IsSolved(NewLRCircuit(2), rebalanced, ecc.BN254.ScalarField()); err == nil {
		t.Fatal("circuit accepted rem >= 2^22 (range check bypassed)")
	}
	t.Log("Tampered/oversized remainders rejected ✓")
}

// ─── Clamp branches ──────────────────────────────────────────

// TestClampSaturation: model outputs far outside the table window [-10, +10]
// must take the clamp branches and yield the saturated sigmoid values.
func TestClampSaturation(t *testing.T) {
	cases := []struct {
		name   string
		bias   float64 // with w=0 the bias is the whole z
		wantLo bool    // expect saturation toward 0 (else toward 1)
	}{
		{"z=+50 saturates high", 50.0, false},
		{"z=-50 saturates low", -50.0, true},
	}

	for _, tc := range cases {
		w, b, x, zt, rem, y := computeWitness(0, 0, tc.bias, 170, 750)
		assignment := newAssignment(w, b, x, zt, rem, y)
		if err := test.IsSolved(NewLRCircuit(2), assignment, ecc.BN254.ScalarField()); err != nil {
			t.Errorf("%s: clamped witness rejected: %v", tc.name, err)
			continue
		}

		yFloat := float64(y.Int64()) / float64(1<<OutputPrecision)
		if tc.wantLo && yFloat > 0.001 {
			t.Errorf("%s: expected y≈0, got %.6f", tc.name, yFloat)
		}
		if !tc.wantLo && yFloat < 0.999 {
			t.Errorf("%s: expected y≈1, got %.6f", tc.name, yFloat)
		}
		t.Logf("%s: y=%.6f ✓", tc.name, yFloat)
	}
}
