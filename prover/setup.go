// Package prover handles the ZK setup phase, proof generation, and verification.
//
// It imports the circuit package for constraint system definitions and provides:
//   - Setup:    CompileCircuit → GenerateKeys → RunSetup
//   - Predict:  ComputeWitness → Prove → Verify
//   - Batch:    RunBatchSetup → BatchPredictParallel
//   - Key I/O:  Save/Load proving and verification keys
package prover

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// ─── Setup Result ────────────────────────────────────────────

// SetupResult holds all artifacts produced by the setup phase.
type SetupResult struct {
	ConstraintSystem constraint.ConstraintSystem
	ProvingKey       plonk.ProvingKey
	VerificationKey  plonk.VerifyingKey

	NumConstraints int
	NumVariables   int
	CompileTime    time.Duration
	SRSTime        time.Duration
	SetupTime      time.Duration
	PKSizeBytes    int
	VKSizeBytes    int
}

// ─── Compile ─────────────────────────────────────────────────

// CompileCircuit compiles the LRCircuit into a Sparse Constraint System.
func CompileCircuit() (constraint.ConstraintSystem, time.Duration, error) {
	start := time.Now()
	c := &circuit.LRCircuit{}
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	elapsed := time.Since(start)
	if err != nil {
		return nil, elapsed, fmt.Errorf("circuit compilation failed: %w", err)
	}
	return ccs, elapsed, nil
}

// ─── SRS + Key Generation ────────────────────────────────────

// GenerateKeys generates the SRS and derives PLONK proving/verification keys.
func GenerateKeys(ccs constraint.ConstraintSystem) (plonk.ProvingKey, plonk.VerifyingKey, time.Duration, error) {
	start := time.Now()
	srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
	if err != nil {
		return nil, nil, time.Since(start), fmt.Errorf("SRS generation failed: %w", err)
	}
	pk, vk, err := plonk.Setup(ccs, srs, srsLagrange)
	elapsed := time.Since(start)
	if err != nil {
		return nil, nil, elapsed, fmt.Errorf("PLONK setup failed: %w", err)
	}
	return pk, vk, elapsed, nil
}

// ─── Full Setup ──────────────────────────────────────────────

// RunSetup executes the complete setup phase: compile → SRS → keys.
func RunSetup() (*SetupResult, error) {
	result := &SetupResult{}

	ccs, compileTime, err := CompileCircuit()
	if err != nil {
		return nil, err
	}
	result.ConstraintSystem = ccs
	result.CompileTime = compileTime
	result.NumConstraints = ccs.GetNbConstraints()

	pk, vk, setupTime, err := GenerateKeys(ccs)
	if err != nil {
		return nil, err
	}
	result.ProvingKey = pk
	result.VerificationKey = vk
	result.SetupTime = setupTime

	var pkBuf, vkBuf bytes.Buffer
	pk.WriteTo(&pkBuf)
	vk.WriteTo(&vkBuf)
	result.PKSizeBytes = pkBuf.Len()
	result.VKSizeBytes = vkBuf.Len()

	return result, nil
}

// ─── Key Serialization ───────────────────────────────────────

// SaveProvingKey serializes the proving key to a file.
func SaveProvingKey(pk plonk.ProvingKey, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()
	_, err = pk.WriteTo(f)
	if err != nil {
		return fmt.Errorf("write proving key: %w", err)
	}
	return nil
}

// SaveVerificationKey serializes the verification key to a file.
func SaveVerificationKey(vk plonk.VerifyingKey, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()
	_, err = vk.WriteTo(f)
	if err != nil {
		return fmt.Errorf("write verification key: %w", err)
	}
	return nil
}

// LoadProvingKey deserializes a proving key from a file.
func LoadProvingKey(path string) (plonk.ProvingKey, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()
	pk := plonk.NewProvingKey(ecc.BN254)
	_, err = pk.(io.ReaderFrom).ReadFrom(f)
	if err != nil {
		return nil, fmt.Errorf("read proving key: %w", err)
	}
	return pk, nil
}

// LoadVerificationKey deserializes a verification key from a file.
func LoadVerificationKey(path string) (plonk.VerifyingKey, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()
	vk := plonk.NewVerifyingKey(ecc.BN254)
	_, err = vk.(io.ReaderFrom).ReadFrom(f)
	if err != nil {
		return nil, fmt.Errorf("read verification key: %w", err)
	}
	return vk, nil
}
