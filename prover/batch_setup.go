// batch_setup.go — Batch circuit compilation and key generation.
package prover

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	kzggeneric "github.com/consensys/gnark-crypto/kzg"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// OutputMode selects what the proof reveals per sample.
type OutputMode string

const (
	// ModeProb exposes the full-precision probability (2^16 scale) per sample.
	// WARNING: exact probabilities allow model extraction from ~d+1 observations.
	ModeProb OutputMode = "prob"
	// ModeLabel exposes only the class bit per sample (default).
	ModeLabel OutputMode = "label"
)

// ParseOutputMode validates a CLI mode string.
func ParseOutputMode(s string) (OutputMode, error) {
	switch OutputMode(s) {
	case ModeProb, ModeLabel:
		return OutputMode(s), nil
	default:
		return "", fmt.Errorf("invalid mode %q (want %q or %q)", s, ModeLabel, ModeProb)
	}
}

// BatchSetupResult holds setup artifacts for batch proving.
type BatchSetupResult struct {
	ConstraintSystem constraint.ConstraintSystem
	ProvingKey       plonk.ProvingKey
	VerificationKey  plonk.VerifyingKey

	BatchSize      int
	NumFeatures    int
	Mode           OutputMode
	NumConstraints int
	CompileTime    time.Duration
	SetupTime      time.Duration
	PKSizeBytes    int
	VKSizeBytes    int
}

// RunBatchSetup compiles the batch circuit and generates keys.
func RunBatchSetup(batchSize int, numFeatures int, mode OutputMode) (*BatchSetupResult, error) {
	return RunBatchSetupFull(batchSize, numFeatures, mode, "results", "")
}

// RunBatchSetupCached compiles the batch circuit, loading keys from disk cache if available.
func RunBatchSetupCached(batchSize int, numFeatures int, mode OutputMode, cacheDir string) (*BatchSetupResult, error) {
	return RunBatchSetupFull(batchSize, numFeatures, mode, cacheDir, "")
}

// RunBatchSetupFull compiles the batch circuit, loading keys from disk cache if available.
// Cache files are keyed by batch size, feature count and output mode:
// results/batch_pk_b{N}_f{F}_{mode}.key (prefixed "ceremony_" when srsPath is set,
// so keys from a real SRS never mix with unsafekzg dev keys).
// If srsPath is non-empty, the SRS is loaded from that file (see LoadCeremonySRS)
// instead of being generated with unsafekzg.
// On cache hit: only circuit compilation (~500ms) is needed instead of full SRS generation.
func RunBatchSetupFull(batchSize int, numFeatures int, mode OutputMode, cacheDir string, srsPath string) (*BatchSetupResult, error) {
	result := &BatchSetupResult{BatchSize: batchSize, NumFeatures: numFeatures, Mode: mode}

	keyPrefix := ""
	if srsPath != "" {
		keyPrefix = "ceremony_"
	}
	pkPath := fmt.Sprintf("%s/%sbatch_pk_b%d_f%d_%s.key", cacheDir, keyPrefix, batchSize, numFeatures, mode)
	vkPath := fmt.Sprintf("%s/%sbatch_vk_b%d_f%d_%s.key", cacheDir, keyPrefix, batchSize, numFeatures, mode)

	// Always compile — fast (~500ms) and required for witness/prove
	var circ frontend.Circuit
	switch mode {
	case ModeLabel:
		circ = circuit.NewBatchLabelCircuit(batchSize, numFeatures)
	case ModeProb:
		circ = circuit.NewBatchCircuit(batchSize, numFeatures)
	default:
		return nil, fmt.Errorf("invalid output mode %q", mode)
	}

	start := time.Now()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circ)
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
	var (
		srsBuf      kzggeneric.SRS
		srsLagrange kzggeneric.SRS
	)
	if srsPath != "" {
		fmt.Printf("    Loading ceremony SRS from %s\n", srsPath)
		srsBuf, srsLagrange, err = LoadCeremonySRS(srsPath, ccs)
		if err != nil {
			return nil, fmt.Errorf("ceremony SRS load failed: %w", err)
		}
	} else {
		// DEV ONLY: locally generated SRS with known toxic waste.
		// Keys derived from this must never back a production verifier.
		srsBuf, srsLagrange, err = unsafekzg.NewSRS(ccs)
		if err != nil {
			return nil, fmt.Errorf("SRS generation failed: %w", err)
		}
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
