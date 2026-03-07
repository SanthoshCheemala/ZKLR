// predict.go — Witness computation, proof generation, and verification.
package prover

import (
	"bytes"
	"fmt"
	"math/big"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// ─── Prediction Result ───────────────────────────────────────

// PredictionResult holds the output of a single ZK prediction.
type PredictionResult struct {
	Height      int
	Weight      int
	Probability float64
	Prediction  string

	ProofBytes []byte
	ProveTime  time.Duration
	VerifyTime time.Duration
	Verified   bool
}

// ─── Witness Computation ─────────────────────────────────────

// ComputeWitness calculates all circuit inputs for given model weights and 2 features.
func ComputeWitness(w1Float, w2Float, bFloat float64, height, weight int) *circuit.LRCircuit {
	scale := new(big.Int).Set(circuit.ScalingFactor)

	w1Big := new(big.Int).SetInt64(int64(w1Float * float64(scale.Int64())))
	w2Big := new(big.Int).SetInt64(int64(w2Float * float64(scale.Int64())))
	bBig := new(big.Int).SetInt64(int64(bFloat * float64(scale.Int64())))
	
	x1Big := big.NewInt(int64(height))
	x2Big := big.NewInt(int64(weight))

	w1x1 := new(big.Int).Mul(w1Big, x1Big)
	w2x2 := new(big.Int).Mul(w2Big, x2Big)
	wx := new(big.Int).Add(w1x1, w2x2)
	zLinear := new(big.Int).Add(wx, bBig)

	offsetScaled := new(big.Int).Lsh(big.NewInt(circuit.ModelOffset), circuit.Precision)
	zShifted := new(big.Int).Add(zLinear, offsetScaled)

	shiftFactor := new(big.Int).Lsh(big.NewInt(1), circuit.Precision-circuit.InputPrecision)
	zTable := new(big.Int).Div(zShifted, shiftFactor)
	rem := new(big.Int).Rem(zShifted, shiftFactor)

	// Clamp for Y computation
	LowerBound := int64((circuit.ModelOffset - circuit.SigmoidOffset) * (1 << circuit.InputPrecision))
	UpperBound := int64((circuit.ModelOffset + circuit.SigmoidOffset) * (1 << circuit.InputPrecision))

	zTableClamped := new(big.Int).Set(zTable)
	if zTableClamped.Int64() < LowerBound {
		zTableClamped = big.NewInt(LowerBound)
	} else if zTableClamped.Int64() > UpperBound {
		zTableClamped = big.NewInt(UpperBound)
	}

	zIndex := zTableClamped.Int64() - LowerBound

	zFloat := float64(zIndex)/float64(1<<circuit.InputPrecision) - float64(circuit.SigmoidOffset)
	yFloat := circuit.SigmoidFloat(zFloat)
	yBig := new(big.Int).SetInt64(int64(yFloat * float64(1<<circuit.OutputPrecision)))

	maxOut := big.NewInt((1 << circuit.OutputPrecision) - 1)
	if yBig.Cmp(maxOut) > 0 {
		yBig = maxOut
	}

	return &circuit.LRCircuit{
		W: [2]frontend.Variable{w1Big, w2Big}, B: bBig,
		X: [2]frontend.Variable{x1Big, x2Big}, ZTable: zTable, Rem: rem, Y: yBig,
	}
}

// GetProbability returns the sigmoid probability from a circuit assignment.
func GetProbability(c *circuit.LRCircuit) float64 {
	yBig, ok := c.Y.(*big.Int)
	if !ok {
		return 0
	}
	return float64(yBig.Int64()) / float64(1<<circuit.OutputPrecision)
}

func GetPrediction(c *circuit.LRCircuit) string {
	if GetProbability(c) >= 0.5 {
		return "OVERWEIGHT"
	}
	return "NORMAL"
}

// ─── Proof Generation ────────────────────────────────────────

// Prove generates a PLONK proof for the given circuit assignment.
func Prove(ccs constraint.ConstraintSystem, pk plonk.ProvingKey, assignment *circuit.LRCircuit) (plonk.Proof, time.Duration, error) {
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		return nil, 0, fmt.Errorf("witness creation failed: %w", err)
	}
	start := time.Now()
	proof, err := plonk.Prove(ccs, pk, witness)
	elapsed := time.Since(start)
	if err != nil {
		return nil, elapsed, fmt.Errorf("proof generation failed: %w", err)
	}
	return proof, elapsed, nil
}

// ─── Proof Verification ──────────────────────────────────────

// Verify checks a PLONK proof using only the verification key and public inputs.
func Verify(vk plonk.VerifyingKey, proof plonk.Proof, assignment *circuit.LRCircuit) (time.Duration, error) {
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		return 0, fmt.Errorf("witness creation failed: %w", err)
	}
	publicWitness, err := witness.Public()
	if err != nil {
		return 0, fmt.Errorf("public witness extraction failed: %w", err)
	}
	start := time.Now()
	err = plonk.Verify(proof, vk, publicWitness)
	elapsed := time.Since(start)
	if err != nil {
		return elapsed, fmt.Errorf("proof verification failed: %w", err)
	}
	return elapsed, nil
}

// ─── Full Prediction Pipeline ────────────────────────────────

// Predict runs the full ZK prediction pipeline for a single sample.
func Predict(setup *SetupResult, w1Float, w2Float, bFloat float64, height, weight int) (*PredictionResult, error) {
	assignment := ComputeWitness(w1Float, w2Float, bFloat, height, weight)

	result := &PredictionResult{
		Height:      height,
		Weight:      weight,
		Probability: GetProbability(assignment),
		Prediction:  GetPrediction(assignment),
	}

	proof, proveTime, err := Prove(setup.ConstraintSystem, setup.ProvingKey, assignment)
	if err != nil {
		return nil, fmt.Errorf("h=%d, w=%d: prove failed: %w", height, weight, err)
	}
	result.ProveTime = proveTime

	var proofBuf bytes.Buffer
	proof.WriteTo(&proofBuf)
	result.ProofBytes = proofBuf.Bytes()

	verifyTime, err := Verify(setup.VerificationKey, proof, assignment)
	result.VerifyTime = verifyTime
	result.Verified = (err == nil)

	if err != nil {
		return result, fmt.Errorf("h=%d, w=%d: verify failed: %w", height, weight, err)
	}

	return result, nil
}
