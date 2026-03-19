// batch_setup.go — Batch circuit compilation and key generation.
package prover

import (
	"bytes"
	"fmt"
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
	return RunBatchSetupCached(batchSize, numFeatures, "results")
}

// RunBatchSetupCached compiles the batch circuit, loading keys from disk cache if available.
// Cache files are keyed by batchSize and numFeatures: results/batch_pk_b{N}_f{F}.key
// On cache hit: only circuit compilation (~500ms) is needed instead of full SRS generation (~23s).
func RunBatchSetupCached(batchSize int, numFeatures int, cacheDir string) (*BatchSetupResult, error) {
	result := &BatchSetupResult{BatchSize: batchSize}

	pkPath := fmt.Sprintf("%s/batch_pk_b%d_f%d.key", cacheDir, batchSize, numFeatures)
	vkPath := fmt.Sprintf("%s/batch_vk_b%d_f%d.key", cacheDir, batchSize, numFeatures)

	// Always compile — fast (~500ms) and required for witness/prove
	start := time.Now()
	c := circuit.NewBatchCircuit(batchSize, numFeatures)
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	if err != nil {
		return nil, fmt.Errorf("batch compile failed: %w", err)
	}
	result.CompileTime = time.Since(start)
	result.ConstraintSystem = ccs
	result.NumConstraints = ccs.GetNbConstraints()

	// Try loading cached keys
	if _, err := os.Stat(pkPath); err == nil {
		if _, err := os.Stat(vkPath); err == nil {
			loadStart := time.Now()
			pk, err := LoadProvingKey(pkPath)
			if err == nil {
				vk, err := LoadVerificationKey(vkPath)
				if err == nil {
					result.ProvingKey = pk
					result.VerificationKey = vk
					result.SetupTime = time.Since(loadStart)

					var pkBuf, vkBuf bytes.Buffer
					pk.WriteTo(&pkBuf)
					vk.WriteTo(&vkBuf)
					result.PKSizeBytes = pkBuf.Len()
					result.VKSizeBytes = vkBuf.Len()
					fmt.Printf("    [cache hit] Loaded keys from %s\n", cacheDir)
					return result, nil
				}
			}
		}
	}

	// Cache miss — full SRS + key generation
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, fmt.Errorf("create cache dir %q: %w", cacheDir, err)
	}
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

	// Save to cache
	if err := SaveProvingKey(pk, pkPath); err != nil {
		return nil, fmt.Errorf("save proving key: %w", err)
	}
	if err := SaveVerificationKey(vk, vkPath); err != nil {
		return nil, fmt.Errorf("save verification key: %w", err)
	}
	fmt.Printf("    [cache saved] Keys written to %s\n", cacheDir)

	return result, nil
}
