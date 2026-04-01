# Smart Contract Integration Guide

## Overview

Your ZKLR Go codebase generates **zero-knowledge proofs** for ML predictions. The smart contract verifies these proofs on-chain without revealing the model weights.

This document explains how to integrate ZKLR with your Solidity smart contract for the ML marketplace.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Contract                            │
│  (Verifies proofs, manages escrow, payment)                  │
└─────────────────────────────────────────────────────────────┘
                ▲                           ▲
                │                           │
         (1) Proof Submission       (2) Verification
                │                           │
┌──────────────────────────┐   ┌────────────────────────────┐
│  ZKLR Prover (Go)        │   │  Verification Key (VK)     │
│  - Private Model Weights │   │  - Exported as Solidity    │
│  - Private Proving Key   │   │  - Deployed to Contract    │
│  - Generates Proofs      │   │  - Verifies All Proofs     │
└──────────────────────────┘   └────────────────────────────┘
```

---

## Step 1: Export Verification Key as Solidity Contract

The seller (prover) generates a Solidity verifier contract from their VK.

**In Go:**
```go
import "github.com/santhoshcheemala/ZKLR/prover"

// Load verification key (generated during setup)
setup, _ := prover.RunBatchSetup(80, 4)
vk := setup.VerificationKey

// Export as Solidity contract
err := prover.ExportVerificationKey(vk, "Verifier.sol")
// Output: Solidity contract that can verify any proof from this prover
```

**Output file: `Verifier.sol`**
- Contains the `verifyProof(bytes proof, uint256[] publicInputs)` function
- Can be deployed to any EVM chain (Ethereum, Polygon, Arbitrum, etc.)
- Only needs VK; does NOT reveal model weights

---

## Step 2: Deploy Verifier to Blockchain

Deploy the `Verifier.sol` contract to your chosen network:

```solidity
// Your marketplace contract inherits from Verifier
pragma solidity ^0.8.0;
import "./Verifier.sol";

contract MLMarketplace is Verifier {
    // Your marketplace logic here
}
```

---

## Step 3: Buyer Initiates Prediction Request

Buyer submits features to the smart contract:

```solidity
function requestPrediction(uint256[4] memory features) external {
    // Store request on-chain
    // Emit event for seller to listen
    emit PredictionRequested(msg.sender, features);
}
```

---

## Step 4: Seller Generates Proof (Off-Chain)

Seller receives the request and generates a ZK proof using ZKLR:

**In Go (seller's backend):**
```go
import "github.com/santhoshcheemala/ZKLR/prover"

// Setup (done once, keys cached)
setup, _ := prover.RunBatchSetup(80, 4)

// Load model weights (PRIVATE - seller keeps these secret)
weights := []float64{0.0264, 0.0230, 0.0240, 0.0243}
bias := -4.894930414628542

// Features from buyer
features := []int{70, 58, 5, 65} // Example

// Generate proof
results := prover.BatchPredictParallel(
    setup, 
    weights, 
    bias, 
    [][]int{features}, 
    4, 0,
)

proof := results[0].ProofBytes
prediction := results[0].Predictions[0].Probability
```

---

## Step 5: Export Proof as JSON for Smart Contract

Convert proof to JSON format for blockchain submission:

```go
// Create witness for public inputs
witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())

// Export proof to JSON
err := prover.ExportProofJSON(proof, publicWitness, "proof.json")
```

**Output: `proof.json`**
```json
{
  "proof": "0xabcd1234...",
  "public_inputs": [
    "70",      // feature 1 (X1)
    "58",      // feature 2 (X2)
    "5",       // feature 3 (X3)
    "65",      // feature 4 (X4)
    "12345"    // prediction probability (Y)
  ],
  "verification_key": "0x7f3e2d1c"
}
```

---

## Step 6: Buyer Submits Proof to Smart Contract

Frontend submits the proof JSON to the contract:

```solidity
function submitPredictionProof(
    bytes calldata proof,
    uint256[4] calldata features,
    uint256 prediction
) external {
    // Prepare public inputs (must match ZKLR format)
    uint256[] memory publicInputs = new uint256[](5);
    publicInputs[0] = features[0];
    publicInputs[1] = features[1];
    publicInputs[2] = features[2];
    publicInputs[3] = features[3];
    publicInputs[4] = prediction;
    
    // Verify proof on-chain using Verifier contract
    require(verifyProof(proof, publicInputs), "Invalid proof");
    
    // If valid: unlock payment, grant access
    // Payments handled by smart contract escrow
}
```

---

## Data Formats

### Public Inputs (What Contract Sees)

The contract receives **5 public inputs**:
1. `X1` - Feature 1 (e.g., height)
2. `X2` - Feature 2 (e.g., weight)
3. `X3` - Feature 3
4. `X4` - Feature 4
5. `Y` - Prediction probability (0-100 or similar range)

**Secret (Never Revealed):**
- `W` - Model weights (weight1, weight2, weight3, weight4)
- `B` - Model bias
- `PK` - Proving key

This is the power of ZK: Prove the model produces valid predictions without revealing the model.

### Proof Format

- **Size:** 584 bytes (constant, always same size)
- **Encoding:** Hex string prefixed with `0x`
- **Platform:** BN254 elliptic curve PLONK

---

## Integration Checklist

- [ ] Export `Verifier.sol` from ZKLR VK
- [ ] Deploy `Verifier.sol` to blockchain
- [ ] Create marketplace contract inheriting `Verifier`
- [ ] Add `requestPrediction(features)` function
- [ ] Add `submitPredictionProof(proof, features, prediction)` function
- [ ] Parse JSON proof from ZKLR backend
- [ ] Extract proof bytes and public inputs
- [ ] Call `verifyProof()` in contract
- [ ] Handle payment/escrow on verified predictions
- [ ] Test end-to-end flow on testnet

---

## Example Flow

1. **Seller** runs Go code:
   ```bash
   go run ./cmd/batch_predict -dataset=data/test_200.csv
   # Generates proof locally using private weights
   ```

2. **Seller** exports proof:
   ```go
   prover.ExportProofJSON(proof, witness, "proof.json")
   ```

3. **Buyer** submits to contract:
   ```js
   // Frontend code
   contract.submitPredictionProof(
       proof.proof,
       proof.public_inputs.slice(0, 4),  // features
       proof.public_inputs[4]             // prediction
   );
   ```

4. **Smart Contract** verifies:
   ```solidity
   verifyProof(proof, publicInputs) // Returns true/false
   ```

5. **Result:** Buyer trusts prediction without ever seeing model weights ✓

---

## Security Notes

- **Private Weights:** Go prover keeps model weights secret. Only VK is public.
- **Immutable Proofs:** Once verified on-chain, buyer has cryptographic proof seller didn't lie.
- **No Replay:** Each prediction generates a unique proof (though same inputs produce same proof).
- **On-Chain Verification:** Smart contract is the source of truth; buyer doesn't need to trust seller's off-chain verification.

---

## Support

For questions:
- ZKLR Go code: See `prover/export.go`
- Solidity interface: See generated `Verifier.sol`
- Example contract: `prover/ContractIntegrationExample()`