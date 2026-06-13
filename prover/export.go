// export.go — Smart Contract Integration
// Exports verification keys as Solidity contracts for blockchain deployment.
package prover

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"path/filepath"

	"github.com/consensys/gnark/backend/plonk"
)

// ExportProofArtifacts writes everything an independent verifier needs:
//
//	dir/vk.key          — verification key
//	dir/manifest.txt    — mode, batch size, features, model commitment
//	dir/batch_NNN.proof — PLONK proof per batch
//	dir/batch_NNN.pubw  — serialized public witness per batch
//
// Verify with: go run ./cmd/verify -dir=<dir>
func ExportProofArtifacts(dir string, setup *BatchSetupResult, results []*BatchPredResult, commitment *big.Int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create export dir: %w", err)
	}
	if err := SaveVerificationKey(setup.VerificationKey, filepath.Join(dir, "vk.key")); err != nil {
		return err
	}

	manifest := fmt.Sprintf(
		"mode: %s\nbatch_size: %d\nnum_features: %d\nmodel_commitment: %s\nmodel_commitment_hex: 0x%s\nbatches: %d\n",
		setup.Mode, setup.BatchSize, setup.NumFeatures,
		commitment.String(), commitment.Text(16), len(results),
	)
	if err := os.WriteFile(filepath.Join(dir, "manifest.txt"), []byte(manifest), 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	exported := 0
	for _, br := range results {
		if br.Error != nil || len(br.ProofBytes) == 0 || len(br.PublicWitnessBytes) == 0 {
			continue
		}
		proofPath := filepath.Join(dir, fmt.Sprintf("batch_%03d.proof", br.BatchIndex))
		pubwPath := filepath.Join(dir, fmt.Sprintf("batch_%03d.pubw", br.BatchIndex))
		if err := os.WriteFile(proofPath, br.ProofBytes, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", proofPath, err)
		}
		if err := os.WriteFile(pubwPath, br.PublicWitnessBytes, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", pubwPath, err)
		}
		exported++
	}
	fmt.Printf("    Exported %d proof(s) + public witnesses + vk → %s\n", exported, dir)
	return nil
}

// ProofJSON wraps a proof with public inputs for smart contract submission.
type ProofJSON struct {
	Proof        string   `json:"proof"`         // hex-encoded proof
	PublicInputs []string `json:"public_inputs"` // [X1, X2, X3, X4, Y]
}

// ExportVerificationKey exports the PLONK verification key as a Solidity contract.
// The contract can verify any proof from this setup on-chain.


// NOTE: For exporting VK as Solidity, use gnark's CLI tools:
//   gnark export-solidity -vk=vk.key -o contracts/ -p verifier -std plonk
// Or use the cmd/export tool which handles this via gnark directly.
func ExportVerificationKey(vk plonk.VerifyingKey, outputPath string) error {
	return fmt.Errorf("use gnark CLI or cmd/export tool for Solidity generation")
}

// ProofToHex converts a proof to hex format for smart contract submission.
func ProofToHex(proofBytes []byte) string {
	return "0x" + hex.EncodeToString(proofBytes)
}

// GetVerificationKeyHash returns a small identifier for the VK.
func GetVerificationKeyHash(vk plonk.VerifyingKey) (string, error) {
	var vkBuf bytes.Buffer
	if _, err := vk.WriteTo(&vkBuf); err != nil {
		return "", fmt.Errorf("serialize vk: %w", err)
	}
	vkBytes := vkBuf.Bytes()
	if len(vkBytes) < 8 {
		return "", fmt.Errorf("vk too small")
	}
	return "0x" + hex.EncodeToString(vkBytes[:8]), nil
}
