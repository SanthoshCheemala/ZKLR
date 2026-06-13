// multiclass.go — Witness construction for the one-vs-rest batch circuit.
package prover

import (
	"math/big"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// ComputeOneVsRestWitness fills an OneVsRestBatchCircuit assignment.
// wFloat is [class][feature]; bFloat is [class]. Samples beyond len(features)
// are padded with a safe dummy row.
func ComputeOneVsRestWitness(wFloat [][]float64, bFloat []float64, features [][]int, batchSize int) *circuit.OneVsRestBatchCircuit {
	numClasses := len(wFloat)
	numFeatures := len(wFloat[0])

	assignment := circuit.NewOneVsRestBatchCircuit(batchSize, numClasses, numFeatures)

	// Scale all classifiers once; commitment covers every class.
	wScaled := make([][]*big.Int, numClasses)
	bScaled := make([]*big.Int, numClasses)
	for c := 0; c < numClasses; c++ {
		wScaled[c], bScaled[c] = ScaleModel(wFloat[c], bFloat[c])
		for j, w := range wScaled[c] {
			assignment.W[c][j] = w
			_ = j
		}
		assignment.B[c] = bScaled[c]
	}
	assignment.Commitment = circuit.ComputeCommitmentMulti(wScaled, bScaled)

	dummy := make([]int, numFeatures)
	for j := range dummy {
		dummy[j] = 100
	}

	for i := 0; i < batchSize; i++ {
		row := dummy
		if i < len(features) {
			row = features[i]
		}
		for j := 0; j < numFeatures; j++ {
			assignment.X[i][j] = big.NewInt(int64(row[j]))
		}
		// Per-class single witnesses give ZTable/Rem/label for this sample.
		for c := 0; c < numClasses; c++ {
			single := ComputeWitness(wFloat[c], bFloat[c], row)
			assignment.ZTable[i][c] = single.ZTable
			assignment.Rem[i][c] = single.Rem
			assignment.Label[i][c] = labelOf(single)
		}
	}
	return assignment
}
