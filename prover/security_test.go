// security_test.go — Batch-level negative tests (tamper matrix, batch circuits).
//
// Uses test.IsSolved (constraint solving, no PLONK setup) so the whole suite
// runs in seconds while still exercising the exact constraint systems.
package prover

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

var (
	secWeights = []float64{testW1, testW2}
	secBias    = testB
	secBatch   = [][]int{{150, 400}, {160, 500}, {170, 700}, {180, 900}, {190, 1000}}
)

// ─── Batch prob circuit ──────────────────────────────────────

func TestBatchValidWitness(t *testing.T) {
	assignment := ComputeBatchWitness(secWeights, secBias, secBatch, len(secBatch))
	err := test.IsSolved(circuit.NewBatchCircuit(len(secBatch), 2), assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("valid batch witness rejected: %v", err)
	}
}

// TestBatchTamperedYRejected: flipping a single Y inside a batch must
// invalidate the whole proof.
func TestBatchTamperedYRejected(t *testing.T) {
	assignment := ComputeBatchWitness(secWeights, secBias, secBatch, len(secBatch))
	assignment.Y[2] = big.NewInt(12345) // tamper one of five outputs

	err := test.IsSolved(circuit.NewBatchCircuit(len(secBatch), 2), assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("batch circuit accepted a tampered Y")
	}
	t.Logf("Correctly rejected tampered Y in batch: %v", err)
}

// TestBatchWrongCommitmentRejected: the batch proof must be bound to the
// published model commitment.
func TestBatchWrongCommitmentRejected(t *testing.T) {
	assignment := ComputeBatchWitness(secWeights, secBias, secBatch, len(secBatch))
	assignment.Commitment = big.NewInt(42)

	err := test.IsSolved(circuit.NewBatchCircuit(len(secBatch), 2), assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("batch circuit accepted a wrong commitment")
	}
	t.Logf("Correctly rejected wrong batch commitment: %v", err)
}

// ─── Batch label circuit ─────────────────────────────────────

func TestBatchLabelValidWitness(t *testing.T) {
	assignment := ComputeBatchLabelWitness(secWeights, secBias, secBatch, len(secBatch))
	err := test.IsSolved(circuit.NewBatchLabelCircuit(len(secBatch), 2), assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("valid label witness rejected: %v", err)
	}

	// Sanity: labels match the prover-side probabilities
	singles := computeBatchSingles(secWeights, secBias, secBatch, len(secBatch))
	for i := range secBatch {
		wantLabel := 0
		if GetProbability(singles[i]) >= 0.5 {
			wantLabel = 1
		}
		if assignment.Label[i] != wantLabel {
			t.Errorf("sample %d: label %v != expected %d (prob %.4f)",
				i, assignment.Label[i], wantLabel, GetProbability(singles[i]))
		}
	}
}

// TestBatchLabelFlippedRejected: flipping one class bit must invalidate the
// proof — the core soundness property of label-only mode.
func TestBatchLabelFlippedRejected(t *testing.T) {
	assignment := ComputeBatchLabelWitness(secWeights, secBias, secBatch, len(secBatch))
	flipped := 1 - assignment.Label[0].(int)
	assignment.Label[0] = flipped

	err := test.IsSolved(circuit.NewBatchLabelCircuit(len(secBatch), 2), assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("label circuit accepted a flipped label")
	}
	t.Logf("Correctly rejected flipped label: %v", err)
}

// TestCommitmentDeterministic: the off-circuit commitment must be stable
// across calls and sensitive to every weight and the bias.
func TestCommitmentDeterministic(t *testing.T) {
	c1 := ModelCommitment(secWeights, secBias)
	c2 := ModelCommitment(secWeights, secBias)
	if c1.Cmp(c2) != 0 {
		t.Fatal("commitment is not deterministic")
	}
	if ModelCommitment([]float64{testW1 + 1e-6, testW2}, secBias).Cmp(c1) == 0 {
		t.Error("commitment insensitive to weight change")
	}
	if ModelCommitment(secWeights, secBias+1e-6).Cmp(c1) == 0 {
		t.Error("commitment insensitive to bias change")
	}
	t.Logf("Model commitment: 0x%s", c1.Text(16))
}
