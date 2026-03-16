// batch_setup.go — Batch circuit compilation and key generation.
package prover

import (
	"bytes"
	"fmt"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// BatchSetupResult holds setup artifacts for batch proving.
type BatchSetupResult struct {
	ConstraintSystem constraint.ConstraintSystem
	ProvingKey       plonk.ProvingKey
	VerificationKey  plonk.VerifyingKey

	BatchSize      int
	NumConstraints int
	CompileTime    time.Duration
	SetupTime      time.Duration
	PKSizeBytes    int
	VKSizeBytes    int
}

// RunBatchSetup compiles the batch circuit and generates keys.
func RunBatchSetup(batchSize int, numFeatures int) (*BatchSetupResult, error) {
	result := &BatchSetupResult{BatchSize: batchSize}

	start := time.Now()
	c := circuit.NewBatchCircuit(batchSize, numFeatures)
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	if err != nil {
		return nil, fmt.Errorf("batch compile failed: %w", err)
	}
	result.CompileTime = time.Since(start)
	result.ConstraintSystem = ccs
	result.NumConstraints = ccs.GetNbConstraints()

	start = time.Now()
	srsBuf, srsLagrange, err := unsafekzg.NewSRS(ccs)
	if err != nil {
		return nil, fmt.Errorf("SRS generation failed: %w", err)
	}
	pk, vk, err := plonk.Setup(ccs, srsBuf, srsLagrange)
	if err != nil {
		return nil, fmt.Errorf("PLONK setup failed: %w", err)
	}
	result.SetupTime = time.Since(start)
	result.ProvingKey = pk
	result.VerificationKey = vk

	var pkBuf, vkBuf bytes.Buffer
	pk.WriteTo(&pkBuf)
	vk.WriteTo(&vkBuf)
	result.PKSizeBytes = pkBuf.Len()
	result.VKSizeBytes = vkBuf.Len()

	return result, nil
}
