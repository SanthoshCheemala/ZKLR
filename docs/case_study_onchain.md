# Case Study: Verifiable, Model-Confidential Credit Scoring On-Chain

This case study demonstrates ZKLR in a concrete real-world setting: a smart
contract that verifies logistic-regression credit decisions on-chain while the
scoring model stays confidential. All numbers below are measured on the real
23-feature credit model (UCI "default of credit card clients"), using genuine
PLONK proofs (not mocks), with Foundry `forge test --gas-report`.

## 1. The problem

A lender or credit bureau scores applicants with a proprietary logistic-
regression model. Two requirements are in tension:

- **Confidentiality.** The model weights are a competitive and regulated asset;
  the institution cannot publish them.
- **Accountability.** Regulators and auditors must be able to confirm that every
  decision came from *one fixed, approved model* — no per-applicant tampering,
  no silent model swaps.

Publishing the model satisfies accountability but breaks confidentiality.
Keeping it secret breaks accountability. ZKLR resolves both.

## 2. The ZKLR solution

The institution commits to one model on-chain via an in-circuit MiMC commitment
`C = MiMC(w_1..w_d, b)`. For each applicant it publishes a PLONK proof that *"the
model committed to in C assigns label L to features X"*. The
`ZKLRBatchVerifier` contract checks two things:

1. the PLONK proof is valid for the public inputs, and
2. the trailing public input equals the pinned `modelCommitment`.

A swapped model produces a different commitment and is rejected. The proof
reveals nothing about the weights beyond the published decisions.

**What is hidden:** the model weights `w, b`.
**What is on-chain:** applicant features `X`, the decision `L`, the commitment `C`.

This is *verifiable inference + model integrity*. (Hiding the applicant's
features as well requires committing to `X` as a private witness — a clean
extension noted in §7, not claimed here.)

## 3. Experimental setup

- Model: credit LR, 23 features, label-only mode (`BatchLabelCircuit`).
- Backend: gnark PLONK over BN254; proof is a constant **584 bytes**.
- Verifier: gnark-exported `PlonkVerifier.sol` + `ZKLRBatchVerifier` wrapper.
- Tooling: Foundry `forge 1.7.1`, `solc 0.8.24`, optimizer 200 runs.
- Reproduce: `./scripts/gas_scan.sh` (regenerates proofs, exports the verifier,
  runs the gas report for each batch size).

## 4. Results

### 4.1 Verification gas vs batch size

| Batch B | Public inputs | Verify gas | Gas / decision |
|--------:|--------------:|-----------:|---------------:|
| 1  | 25  | 317,618   | 317,618 |
| 5  | 121 | 446,190   | 89,238  |
| 10 | 241 | 607,365   | 60,736  |
| 20 | 481 | 931,213   | 46,561  |
| 40 | 961 | 1,585,024 | **39,626** |

### 4.2 On-chain cost model

A linear fit against the public-input count is essentially exact:

```
verify_gas = 282,005 + 1,354.5 * (public inputs)      R² = 1.0000
```

- **Fixed core ≈ 282K gas** — the PLONK pairing / commitment-opening checks,
  independent of batch size. This is the dominant cost at small batches.
- **≈ 1,355 gas per public input** — the marginal MSM + calldata term.

Because each sample contributes 24 public inputs (23 features + 1 label), the
per-decision gas asymptotically approaches a floor of `1,355 × 24 ≈ 32,500 gas`.

### 4.3 The key finding: batches win on *both* axes

ZKLR has two independent amortization mechanisms, and they point the same way:

- **Proving** amortizes the shared sigmoid lookup table over the batch; cost is
  minimized when the batch fills the power-of-two FFT domain (optimal at B=80,
  98.7% domain fill → 0.187 s/sample). See `docs/correctness.md` / cost model.
- **On-chain verification** amortizes the fixed ~282K-gas pairing core over the
  batch; per-decision gas falls monotonically (317K → 40K and approaching ~32K).

There is **no prove-vs-verify tradeoff**: larger batches are better for the
prover *and* for the on-chain verifier. The only ceiling is proof generation
favouring the FFT-domain boundary.

### 4.4 Deployment and rejection costs

| Operation | Gas | Note |
|---|---:|---|
| Deploy `ZKLRBatchVerifier` | 2,001,940 | one-time; VK-sized, batch-independent |
| Reject wrong model commitment | 50,223 | reverts *before* the costly pairing |
| Verify batch of 10 | 607,365 | full happy path |

Enforcing model integrity is therefore nearly free: a substituted model is
rejected at ~50K gas, roughly an order of magnitude below a full verification.

### 4.5 Monetary cost

At 607K gas for a batch of 10 decisions (60.7K gas/decision):

| Venue | Gas price | Cost / batch | Cost / decision |
|---|---|---:|---:|
| Ethereum L1 | 20 gwei, $3,000 ETH | ~$36 | ~$3.6 |
| L2 rollup | 0.05 gwei | ~$0.09 | ~$0.009 |

On-chain L1 verification is viable for high-value or audit-anchored decisions;
an L2 makes per-applicant verification cost a fraction of a cent — the realistic
deployment target.

## 5. Security guarantees (measured)

`forge test` exercises four adversarial cases against real proofs, all passing:

- valid proof for the pinned model **verifies**;
- a flipped label **fails** verification;
- a proof presented under a different commitment **reverts** (`WrongModelCommitment`);
- a corrupted proof byte **fails** verification.

These mirror the off-chain tamper matrix in `circuit/security_test.go` and
`prover/security_test.go`.

## 6. Reproducibility

```bash
# one batch size, full report
./scripts/run_gas_benchmark.sh 10 data/credit_gas20.csv "$W" "$B"

# the gas-vs-batch scan in §4.1
./scripts/gas_scan.sh          # → results/gas_scan.csv, results/fig_gas_scan.png
```

## 7. Limitations and extensions

- **Applicant-feature privacy.** Features are currently public inputs (the model
  is the secret). Committing to `X` as a private witness and exposing only a hash
  would hide applicant data too, at the cost of extra constraints.
- **Calldata.** Large batches post many public inputs as calldata; a commit-and-
  prove variant (verify a hash of the public inputs on-chain) would bound
  calldata to O(1) and is the natural next step for very large batches.
