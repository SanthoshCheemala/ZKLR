package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"

	"github.com/santhoshcheemala/ZKLR/prover"
)

func main() {
	batch := flag.Int("batch", 80, "batch size (must match prover)")
	features := flag.Int("features", 4, "number of model features")
	outPrefix := flag.String("out", "Verifier", "output file prefix")
	help := flag.Bool("help", false, "show usage information")
	flag.Parse()

	if *help {
		printHelp()
		return
	}

	exportVK(*batch, *features, *outPrefix)
}

// exportVK generates and exports the verification key for Solidity contract generation.
func exportVK(batchSize, numFeatures int, outPrefix string) {
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println("  ZKLR Verification Key Export Tool")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf("\n[1] Generating Setup (batch=%d, features=%d)...\n", batchSize, numFeatures)

	// Setup: compile circuit and load/regenerate keys
	setup, err := prover.RunBatchSetup(batchSize, numFeatures)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Setup failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("    Constraints: %d\n", setup.NumConstraints)
	fmt.Printf("    Setup time: %v\n", setup.SetupTime)

	// Export verification key to binary file
	fmt.Printf("\n[2] Exporting Verification Key...\n")
	vkPath := outPrefix + ".vk"
	var vkBuf bytes.Buffer
	if _, err := setup.VerificationKey.WriteTo(&vkBuf); err != nil {
		fmt.Fprintf(os.Stderr, "✗ Failed to marshal VK: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(vkPath, vkBuf.Bytes(), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "✗ Failed to write VK file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("    Saved to: %s (%d bytes)\n", vkPath, vkBuf.Len())

	// Get VK hash for reference
	vkHash, _ := prover.GetVerificationKeyHash(setup.VerificationKey)
	fmt.Printf("    VK Hash: %s\n", vkHash)

	// Print next steps
	fmt.Println("\n════════════════════════════════════════════════════════════")
	fmt.Println("  ✓ Export Complete")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println("\n[NEXT STEPS]")
	fmt.Println("")
	fmt.Println("  1. Generate Solidity contract using gnark CLI:")
	fmt.Printf("     gnark export-solidity -vk=%s -o contracts/ -p Verifier\n", vkPath)
	fmt.Println("")
	fmt.Println("  2. Deploy Verifier.sol to blockchain:")
	fmt.Println("     truffle migrate --network polygon")
	fmt.Println("")
	fmt.Println("  3. In your marketplace contract, inherit from Verifier:")
	fmt.Println("     import './Verifier.sol';")
	fmt.Println("     contract MLMarketplace is Verifier { ... }")
	fmt.Println("")
	fmt.Println("  4. To verify proofs on-chain:")
	fmt.Println("     bool valid = verifyProof(proofBytes, [x1, x2, x3, x4, y]);")
	fmt.Println("")
	fmt.Println("[DOCUMENTATION]")
	fmt.Println("  See: docs/SMART_CONTRACT_INTEGRATION.md")
	fmt.Println("")
}

// printHelp prints usage information.
func printHelp() {
	fmt.Print(`
ZKLR Verification Key Export Tool
Export VK for generating Solidity verifier contracts

USAGE:
  go run ./cmd/export/ [options]

OPTIONS:
  -batch=N       Batch size [default: 80]
  -features=N    Model features [default: 4]
  -out=PREFIX    Output file prefix [default: Verifier]
  -help          Show this message

EXAMPLES:

  Default (batch=80, features=4):
    go run ./cmd/export/ -out=Verifier

  Custom setup:
    go run ./cmd/export/ -batch=20 -features=2 -out=CustomVK

WORKFLOW:

  1. Go prover sets up with hardcoded defaults:
     go run ./cmd/batch_predict -dataset=data/test_200.csv
     → Uses: batch=80, workers=auto, features=4 (hardcoded)
     → Generates proofs using private model weights

  2. Export VK for smart contract:
     go run ./cmd/export/ -out=Verifier
     → Generates: Verifier.vk (binary)

  3. Generate Solidity contract:
     gnark export-solidity -vk=Verifier.vk -o contracts/ -p Verifier
     → Generates: Verifier.sol (Solidity contract)

  4. Deploy to blockchain:
     truffle migrate --network ethereum
     → Smart contract can now verify any proof from this setup

  5. Use in marketplace:
     JavaScript/Web3: contract.verifyProof(proof, [x1, x2, x3, x4, y])
     → Returns: boolean (proof valid)

SECURITY:

  PublicInputs: [feature1, feature2, feature3, feature4, prediction]
  - Visible on-chain
  - Anyone can see the prediction for the given features
  
  Secret (never revealed):
  - Model weights (W1, W2, W3, W4)
  - Model bias (B)
  - Proving key (PK)
  - Training data

FILES:

  Input:  Nothing (uses hardcoded setup)
  Output: Verifier.vk → Use with gnark CLI

REQUIREMENTS:

  gnark CLI (for Solidity export):
    go install github.com/consensys/gnark/cmd/gnark@latest

  Truffle (for deployment):
    npm install -g truffle

See docs/SMART_CONTRACT_INTEGRATION.md for complete guide.
`)
}
