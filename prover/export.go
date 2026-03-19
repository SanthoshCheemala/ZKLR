// export.go — Smart Contract Integration
// Exports verification keys as Solidity contracts for blockchain deployment.
package prover

import (
	"bytes"
	"encoding/hex"
	"fmt"

	"github.com/consensys/gnark/backend/plonk"
)

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
