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

// ComputeWitness calculates all circuit inputs for given model weights and N features.
func ComputeWitness(wFloat []float64, bFloat float64, features []int) *circuit.LRCircuit {
	if len(wFloat) != len(features) {
		panic("model weights and features length mismatch")
	}
	scale := new(big.Int).Set(circuit.ScalingFactor)

	wBig := make([]frontend.Variable, len(wFloat))
	xBig := make([]frontend.Variable, len(features))
	
	for i := 0; i < len(wFloat); i++ {
		wBig[i] = new(big.Int).SetInt64(int64(wFloat[i] * float64(scale.Int64())))
		xBig[i] = big.NewInt(int64(features[i]))
	}

	bBig := new(big.Int).SetInt64(int64(bFloat * float64(scale.Int64())))

	wx := big.NewInt(0)
	for i := 0; i < len(wFloat); i++ {
		wixi := new(big.Int).Mul(wBig[i].(*big.Int), xBig[i].(*big.Int))
		wx.Add(wx, wixi)
	}
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
		W: wBig, B: bBig,
		X: xBig, ZTable: zTable, Rem: rem, Y: yBig,
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
func Predict(setup *SetupResult, wFloat []float64, bFloat float64, features []int) (*PredictionResult, error) {
	assignment := ComputeWitness(wFloat, bFloat, features)

	h, w := 0, 0
	if len(features) > 0 { h = features[0] }
	if len(features) > 1 { w = features[1] }
	
	result := &PredictionResult{
		Height:      h,
		Weight:      w,
		Probability: GetProbability(assignment),
		Prediction:  GetPrediction(assignment),
	}

	proof, proveTime, err := Prove(setup.ConstraintSystem, setup.ProvingKey, assignment)
	if err != nil {
		return nil, fmt.Errorf("h=%d, w=%d: prove failed: %w", h, w, err)
	}
	result.ProveTime = proveTime

	var proofBuf bytes.Buffer
	proof.WriteTo(&proofBuf)
	result.ProofBytes = proofBuf.Bytes()

	verifyTime, err := Verify(setup.VerificationKey, proof, assignment)
	result.VerifyTime = verifyTime
	result.Verified = (err == nil)

	if err != nil {
		return result, fmt.Errorf("h=%d, w=%d: verify failed: %w", h, w, err)
	}

	return result, nil
}
