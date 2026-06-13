// multiclass_test.go — One-vs-rest circuit: positive + tamper tests.
package prover

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// Three toy one-vs-rest classifiers over two features.
var (
	mcWeights = [][]float64{
		{0.02, -0.01},
		{-0.015, 0.025},
		{0.005, 0.005},
	}
	mcBias  = []float64{-1.0, 0.5, -2.0}
	mcBatch = [][]int{{150, 400}, {160, 500}, {170, 700}, {180, 900}}
)

func TestOneVsRestValidWitness(t *testing.T) {
	assignment := ComputeOneVsRestWitness(mcWeights, mcBias, mcBatch, len(mcBatch))
	err := test.IsSolved(
		circuit.NewOneVsRestBatchCircuit(len(mcBatch), len(mcWeights), 2),
		assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("valid one-vs-rest witness rejected: %v", err)
	}

	// Sanity: labels match per-class single-circuit predictions
	for i, row := range mcBatch {
		for c := range mcWeights {
			single := ComputeWitness(mcWeights[c], mcBias[c], row)
			if assignment.Label[i][c] != labelOf(single) {
				t.Errorf("sample %d class %d: label mismatch", i, c)
			}
		}
	}
}

func TestOneVsRestFlippedLabelRejected(t *testing.T) {
	assignment := ComputeOneVsRestWitness(mcWeights, mcBias, mcBatch, len(mcBatch))
	assignment.Label[1][2] = 1 - assignment.Label[1][2].(int)

	err := test.IsSolved(
		circuit.NewOneVsRestBatchCircuit(len(mcBatch), len(mcWeights), 2),
		assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("one-vs-rest circuit accepted a flipped label bit")
	}
	t.Logf("Correctly rejected flipped label: %v", err)
}

func TestOneVsRestWrongCommitmentRejected(t *testing.T) {
	assignment := ComputeOneVsRestWitness(mcWeights, mcBias, mcBatch, len(mcBatch))
	assignment.Commitment = big.NewInt(7)

	err := test.IsSolved(
		circuit.NewOneVsRestBatchCircuit(len(mcBatch), len(mcWeights), 2),
		assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("one-vs-rest circuit accepted a wrong commitment")
	}
	t.Logf("Correctly rejected wrong commitment: %v", err)
}

// TestOneVsRestClassSwapRejected: classifiers cannot be reordered relative to
// the committed class order (the per-class binding property).
func TestOneVsRestClassSwapRejected(t *testing.T) {
	assignment := ComputeOneVsRestWitness(mcWeights, mcBias, mcBatch, len(mcBatch))
	// Swap two classifiers' weights but keep the original commitment + labels
	assignment.W[0], assignment.W[1] = assignment.W[1], assignment.W[0]
	assignment.B[0], assignment.B[1] = assignment.B[1], assignment.B[0]

	err := test.IsSolved(
		circuit.NewOneVsRestBatchCircuit(len(mcBatch), len(mcWeights), 2),
		assignment, ecc.BN254.ScalarField())
	if err == nil {
		t.Fatal("one-vs-rest circuit accepted swapped classifiers under the original commitment")
	}
	t.Logf("Correctly rejected class swap: %v", err)
}
