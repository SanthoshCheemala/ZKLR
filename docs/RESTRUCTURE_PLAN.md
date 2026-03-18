# ZKLR Restructure & Smart Contract Integration Plan

## Goal
Restructure the codebase into a clean prover/verifier separation, remove unnecessary files,
add smart contract export capabilities, and write integration documentation for the teammate
working on the Solidity smart contract + frontend.

---

## Workplan

### Phase 1: Cleanup & Restructure
- [ ] Remove root-level compiled binaries: `batch_predict`, `predict`, `benchmark`, `batch_results.txt`
- [ ] Delete `v1_archive/` entirely (old code, no longer needed)
- [ ] Delete `prover/results/` (misplaced cache keys)
- [ ] Remove duplicate/unused data files:
  - `data/student_dataset.csv`, `data/student_dataset_test.csv`
  - `data/model_weights.txt` (superseded by `data/bmi_model_weights.txt`)
  - `data/synthetic_5f_*.csv`, `data/synthetic_5f_weights.txt` (unused 5-feature experiment)
  - `data/test_small.csv`
- [ ] Remove large binary `.key` files from git tracking — add to `.gitignore`, regenerated at runtime
- [ ] Remove stale result TXTs: `config_sweep.txt`, `prediction_results.txt`, `verification_report.txt`, etc.
- [ ] Clean up `scripts/` — keep only `prepare_model.py` and `generate_bmi_dataset.py`

### Phase 2: Smart Contract Flexibility
- [ ] Add `prover/export.go` — new file with:
  - `ExportSolidityVerifier(vk, outputPath)` — exports gnark VK as `.sol` Solidity contract
  - `ExportProofJSON(proof, publicInputs, outputPath)` — serializes proof + public inputs to JSON for frontend/contract calls
  - `ProofToCalldata(proof, publicWitness)` — returns ABI-encoded calldata bytes ready for `verifyProof()` on-chain
- [ ] Add `cmd/export/main.go` — CLI: `go run ./cmd/export/ -vk=results/verification.key -out=contracts/Verifier.sol`

### Phase 3: Documentation
- [ ] Write `docs/SMART_CONTRACT_INTEGRATION.md`:
  - How to export the Solidity verifier contract
  - What public inputs the contract expects (X features, Y prediction)
  - Proof format and how to pass it from Go → frontend → smart contract
  - Example flow: User submits features → Go prover generates proof → JSON → frontend → contract verifies

### Phase 4: Final
- [ ] Build + test everything
- [ ] Commit and push to main

---

## Key Technical Notes
- gnark v0.11.0 supports `ExportSolidity` for PLONK on BN254 curve (which this project uses) ✅
- The VK (`verification.key`) is the only thing the smart contract needs — PK stays private with the prover
- Public inputs to the contract: `X[]` (features) and `Y` (predicted probability) — model weights W,B stay **secret**
- Proof is **584 bytes constant** — cheap to pass on-chain
- Teammate's smart contract needs: `Verifier.sol` (exported from VK) + proof calldata format
