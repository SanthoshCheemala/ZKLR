# ZKLR → Journal Readiness Plan (Technical Work)

> **Status (2026-06-12):** Phases 0, 1, 4 complete. Phase 2 harness complete
> (gas numbers pending a Foundry run: `make gas`). Phase 3 complete except the
> HPC re-sweep and N≥10 repetition methodology (items 3.6–3.8, need HPC).
> Phase 5 deliverables written: `docs/correctness.md`,
> `docs/comparative_analysis.md`, fidelity + cost-model figures,
> `make reproduce`, CI. Remaining before manuscript: token revocation +
> history rewrite, HPC re-sweep, gas table, ceremony-SRS keys for deployed
> artifacts.

Goal: make ZKLR technically journal-ready — sound protocol, rigorous results and
graphs, correctness proofs, and comparative analysis. (Manuscript writing and venue
logistics are out of scope for this plan.)

Core thesis the results must support:
**"Shared-lookup amortization makes batched ZK inference practical: a sound,
end-to-end (off-chain prove, on-chain verify) pipeline with an analytical cost
model, validated against EZKL."**

Estimated total effort: 6–8 weeks part-time. Phases 0–3 + 5 are the minimum bar;
Phase 4 is the novelty upgrade that meaningfully raises acceptance odds.

---

## Phase 0 — Hygiene (Day 1) 🔴 do first

| # | Task | Where |
|---|------|-------|
| 0.1 | **Revoke the leaked GitHub token** (committed in `data/git.token`, pushed to origin). Then delete the file and rewrite history with `git filter-repo --path data/git.token --invert-paths`. | GitHub settings + repo |
| 0.2 | Fix CLI bug: `-weights` required-check runs *after* parsing, so the documented default invocation crashes with a confusing error. Move the check before `strings.Split`. | `cmd/batch_predict/main.go:51-67` |
| 0.3 | Fix accuracy indexing: predictions are matched to dataset rows by (f1,f2) equality with a silent fallback to row 0. Use `batchIndex*batchSize + i` instead. | `cmd/batch_predict/main.go:152-175` |
| 0.4 | Remove hardcoded metrics: print measured `len(ProofBytes)` instead of literal `584`; remove "speedup vs single" computed from an assumed 2.0 s baseline (replaced by a measured baseline in Phase 3). | `cmd/batch_predict/main.go:215-216` |
| 0.5 | Repo cleanup: delete LaTeX build artifacts (`New Project/report.aux`, `.synctex`, `.fls`, `.bak`, …), remove or implement the `cmd/worker_benchmark` stub, fix stale comments (`sigmoid.go` "32769 entries" / "[0,8]" range; `docs/circuit_architecture.md` offset described as 16 and 10 vs actual 1000). | repo-wide |

**Exit criteria:** token revoked; `go run ./cmd/batch_predict -weights=... ` and the
README agree; no fabricated numbers in program output.

---

## Phase 1 — Protocol soundness (Week 1) 🔴 prerequisite for any submission

The current circuit proves "there exist W, B such that Y = sigmoid(W·X+B)" — W and B
are unbound, so the statement is nearly vacuous. Reviewers will catch this.

1. **Weight commitment (the key fix).**
   - Add `Commitment frontend.Variable \`gnark:",public"\`` to `LRCircuit` and `BatchCircuit`.
   - In `Define`, compute MiMC over all `W[i]` and `B` (`gnark/std/hash/mimc`) and
     `api.AssertIsEqual(c.Commitment, h.Sum())`. Cost ≈ a few hundred constraints —
     negligible vs the ~105K table.
   - Off-circuit, compute the same MiMC (gnark-crypto, BN254 fr) over the scaled
     integer weights and publish it once. Witness fills it in `ComputeWitness` /
     `ComputeBatchWitness`.
   - Statement becomes: "Y = sigmoid(W·X+B) **for the W, B committed to in C**."
2. **Negative tests** (currently missing):
   - wrong commitment rejected; one tampered Y inside a batch of 80 rejected;
     X at the 12-bit boundary (4095) accepted, 4096 rejected; out-of-range z exercises
     both clamp branches.
3. **Real SRS option.** Keep `unsafekzg` for dev behind a flag; add a path that loads a
   ceremony SRS (Aztec Ignition transcript covers BN254 up to 100M points; batch=80
   needs ~2^20). Paper reports numbers with the real SRS.
4. **Independent verifier:** new `cmd/verify` that loads VK + proof + public inputs
   (X, Y, commitment) from disk and verifies — no shared process state with the prover.
   This is the artifact that demonstrates the actual claim.
5. **Label-only output mode (model-extraction defense).** Publishing exact Y at 2^16
   precision lets an observer recover W, B from ~d+1 predictions via
   `logit(Y) = W·X + B` (Tramèr et al., USENIX Sec'16 — LR extraction from
   confidence scores). Add a circuit mode that compares `Y ≥ 0.5` *in-circuit* and
   exposes only the class bit (optionally coarse buckets) as public. Make this the
   default for the paper's privacy claims; keep full-probability mode as an option
   with the leakage explicitly discussed in Limitations.

**Exit criteria:** all tests green including new negative tests; a proof produced by
`cmd/batch_predict` verifies via `cmd/verify` from files alone; commitment mismatch fails.

---

## Phase 2 — On-chain verification, measured (Week 2)

Turns the "smart contract integration design" claim into a result.

1. Export the Solidity verifier (`vk.ExportSolidity` in gnark v0.11 for PLONK/BN254).
2. Deploy on a local Anvil node (Foundry); write a `forge` test that submits a real
   proof + public inputs; record **gas per verification** via `--gas-report`.
3. Pin the weight commitment in the contract (constructor arg) so on-chain
   verification binds to the committed model.
4. Measure and report calldata size honestly: batch=80 × 4 features ⇒ ~401 public
   field elements ≈ 12.8 KB calldata. Discuss the public-input-hashing optimization
   (commit to X,Y in-circuit, pass only the hash on-chain) as future work or implement
   it if time allows — it is itself a publishable measurement ("gas vs batch size").

**Exit criteria:** a table: batch size → proof size, calldata bytes, verification gas,
USD-equivalent at a stated gas price.

---

## Phase 3 — Evaluation upgrade (Weeks 3–4) 🔴 reviewers' main demand

1. **EZKL baseline.** Export the same LR model to ONNX (sklearn → onnx),
   run `ezkl gen-settings / compile-circuit / setup / prove / verify` on the same
   test set and hardware. Compare: proving time/sample, proof size, verify time,
   on-chain gas (EZKL exports an EVM verifier too). Report honestly even where EZKL wins.
2. **Groth16 baseline** (cheap: same gnark code, `backend/groth16` + R1CS builder) —
   gives a second point of comparison for proof size / verify cost vs universal setup.
3. **Measured single-proof baseline.** Run batch=1 end-to-end; all speedup claims
   derive from this measurement, replacing the assumed 2.0 s constant.
4. **Cost model.** Fit `T_prove(B) = a + b·B` (table cost + per-sample logic) to the
   existing 28-point HPC sweep plus new runs; plot predicted vs measured (report R²);
   derive a batch-size/worker selection rule from the model. This converts the heatmap
   into an analytical contribution.
5. **Fixed-point fidelity study on 3 real-world datasets.** Demonstrates ZKLR does not
   alter the LR model's behavior, on data reviewers recognize:
   - **Breast Cancer Wisconsin (Original)** (UCI #15; 683 usable samples, 9 features,
     raw integers 1–10): features fit the 12-bit range check natively — zero input
     quantization error, isolating the effect of weight quantization.
   - **Pima Indians Diabetes** (768 samples, 8 features): canonical LR benchmark;
     features fit 12 bits with simple ×10/×1000 scaling (BMI, pedigree).
   - **Default of Credit Card Clients, Taiwan** (UCI #350; 30,000 samples, 23 features):
     finance domain + real-data scale for the throughput benchmarks; min-max scale
     features to [0, 4000] integers (handles negative bill amounts).
   - Alternates: German Credit (Statlog), Heart Disease Cleveland.

   Protocol per dataset: train float sklearn LR (fixed seed/split) → quantize inputs
   (12-bit) and weights (2^32) → run the full ZK pipeline over the test set → report
   float vs circuit accuracy, prediction agreement %, max/mean |Δp|, AUC delta.
   Verify empirically that |W·X+B| stays within the ±10 table range per model
   (clamped tail is sigmoid-saturated and cannot flip a prediction — state this).
   Expected: ~100% agreement, |Δp| bounded by table resolution (~2^-10).
6. Re-run the HPC sweep with the post-Phase-1 circuit (commitment changes constraint
   count slightly) so every number in the paper comes from one code version. Pin the
   exact commit hash; script everything (`run_hpc_sweep.sh` already exists — extend it).
7. **Benchmark methodology (journal-grade rigor).** N ≥ 10 repetitions per
   configuration; report median ± std and plot error bars; separate warm vs cold
   key-cache numbers; include witness-generation time in the cost model, not just
   `plonk.Prove`. Record exact hardware (CPU model, core count, RAM, OS) and pinned
   Go/gnark versions in the paper's setup section.
8. **Key-pool memory/throughput tradeoff.** Measure peak RSS and throughput across
   key-pool sizes (1, 2, 4, 8, 16) at fixed workers — this is the project's most
   original systems mechanism; present it as a curve, not a sentence. Also fix the
   fallback in `prover/batch_predict.go:163` that silently shares one key on clone
   failure before benchmarking.
9. *(Optional, data partly exists)* Curve ablation: BN254 vs BLS12-377
   (`results/laptop_bls12377_100_run.txt`) — one small table; BN254 justified by EVM
   compatibility.

**Exit criteria:** comparison table (ZKLR-PLONK, ZKLR-Groth16, EZKL) on identical
hardware/model/data; cost-model figure; fidelity table; all numbers reproducible from
scripts in the repo.

---

## Phase 4 — Generalize the amortization claim (Weeks 4–6) 🟡 novelty upgrade

This is what lifts the paper from "we did LR" to "a technique". Pick at least one:

- **Option A (recommended, lowest risk): multi-class one-vs-rest LR.** c classifiers
  share the *same* sigmoid table: cost ≈ table + c·B·logic. Demonstrates the
  amortization scales in a second dimension (classes), validates the cost model again.
- **Option B: one-hidden-layer MLP** (k neurons, sigmoid activations) — all k·B
  activations share one table. Stronger claim, more circuit work (hidden-layer
  truncation per neuron).

Re-run the cost model against the new circuit family; show `T = table + (units)·logic`
holds. Update the paper's claim accordingly.

**Exit criteria:** constraint-count and proving-time scaling plots for the second model
family, matching the cost model's predictions.

---

## Phase 5 — Results, graphs, correctness proofs, comparative analysis (Weeks 6–8)

All numbers from one pinned commit, regenerated by scripts (no hand-copied values —
the current hardcoded 584 B / 2.0 s / 7.1× vs 12.5× inconsistencies must be
impossible by construction).

### 5.1 Results tables (machine-generated from raw result files)
- **T1 — Circuit characteristics:** constraints, PK/VK size, compile/setup time vs
  batch size; with vs without commitment; full-probability vs label-only mode.
- **T2 — HPC sweep:** batch × workers → wall-clock/sample and throughput,
  median ± std over N ≥ 10 runs (extends the existing 28-point sweep).
- **T3 — Key-pool tradeoff:** pool size (1, 2, 4, 8, 16) → peak RSS, throughput.
- **T4 — Fidelity per real dataset** (WBC, Pima, Taiwan credit): float vs circuit
  accuracy, prediction agreement %, max/mean |Δp|, AUC delta.
- **T5 — On-chain costs:** batch size → proof size, calldata bytes, verification gas,
  USD at a stated gas price.
- **T6 — Head-to-head:** ZKLR-PLONK vs ZKLR-Groth16 vs EZKL — prove time/sample,
  proof size, verify time, gas, setup type (universal vs per-circuit).
- **T7** *(if Phase 4 done)* — second model family (multi-class / MLP) scaling.

### 5.2 Graphs (extend `scripts/generate_report_visualizations.py`; one command
regenerates all figures)
- **F1:** constraints & prove time vs batch size — measured points + fitted cost
  model `T(B) = a + b·B`, annotated R² (the amortization claim in one figure).
- **F2:** HPC sweep heatmap (batch × workers) with median values; companion plot
  with error bars.
- **F3:** throughput vs workers per batch size (shows the memory-bandwidth knee).
- **F4:** key-pool size vs peak RSS and throughput (dual-axis tradeoff curve).
- **F5:** speedup vs *measured* batch=1 baseline (replaces the assumed-2.0 s figure).
- **F6:** per dataset — float vs circuit probability scatter + |Δp| histogram
  (the "ZKLR doesn't change the model" figure).
- **F7:** comparative bars — ZKLR vs EZKL vs Groth16 (prove time, proof size, gas).
- **F8:** verification gas vs batch size.

### 5.3 Correctness proofs (`docs/correctness.md`)
- **Formal relation:** precise NP relation R for both modes (committed weights;
  label-only output), with public/private input partition.
- **Gadget soundness arguments:**
  - truncation: unique decomposition of `z = ZTable·2^22 + Rem` given the
    24-bit/22-bit range checks and the bound on z;
  - clamp: out-of-range z lands in the sigmoid-saturated region and cannot flip the
    label (with the explicit bound);
  - commitment: binding of (W, B) via MiMC collision resistance.
- **Protocol properties:** completeness, knowledge soundness, zero-knowledge —
  stated as inherited from PLONK + logderivlookup, with citations.
- **Quantization error bound:** analytic bound on |Δp| from 2^10 index / 2^16 output
  precision; shown alongside the measured max |Δp| from T4 (analytic ≥ measured).
- **Empirical tamper matrix:** table of every witness/public component tampered
  (wrong Y, wrong Z, wrong Rem, wrong commitment, one bad sample in a batch of 80,
  X out of 12-bit range) → all rejected; generated by the Phase 1 negative test
  suite, referenced as evidence.

### 5.4 Comparative analysis (`docs/comparative_analysis.md`)
- **Quantitative:** T6/F7 on identical model, datasets, and hardware; honest
  discussion of where ZKLR wins (amortized prove time, universal setup) and loses
  (e.g., EZKL generality, Groth16 proof size).
- **Qualitative feature matrix vs related work:** ZEN, zkCNN, Mystique, ezDPS,
  zkml (Kang), EZKL, zkLLM — columns: model class, proof system, activation
  handling, batching/amortization, on-chain verification, setup type, code available.
- **Positioning paragraph:** what ZKLR demonstrates that each closest competitor
  does not (the amortization cost model + measured end-to-end on-chain pipeline).

### 5.5 Reproducibility
- Dockerfile + `make reproduce` regenerating every table (5.1) and figure (5.2)
  from scratch; CI workflow (build + `go test ./...` on push); exact hardware and
  pinned Go/gnark versions recorded in the results files.

---

## Dependency graph

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 5
                 │            ↗
                 └→ Phase 3 ──┤
                 └→ Phase 4 ──┘   (4 optional but high-value)
```

## Minimum completion bar
Phases 0–3 + Phase 5. Without the commitment (1.1), label-only mode (1.5),
independent verification (1.4), and the EZKL baseline (3.1), the headline claims
are not supported by the artifacts — everything downstream depends on those four.
