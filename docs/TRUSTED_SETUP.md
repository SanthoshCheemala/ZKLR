# Trusted Setup (SRS)

PLONK needs a universal Structured Reference String (SRS). ZKLR supports two sources:

## 1. Dev mode (default) — NOT for production

Without `-srs`, setup uses gnark's `test/unsafekzg`, which generates an SRS from a
**locally known** secret tau. Anyone who knows tau can forge proofs, so:

- fine for development and performance benchmarking,
- **never** export keys derived from it to a production verifier or smart contract.

Dev keys are cached as `results/batch_pk_b{N}_f{F}_{mode}.key`.

## 2. Ceremony SRS — production

Pass a gnark-serialized BN254 SRS file:

```bash
go run ./cmd/batch_predict -srs=ceremony_bn254.srs -weights=... -bias=...
```

The loader (`prover.LoadCeremonySRS`):
- truncates the transcript to the circuit's required size
  (`NextPowerOfTwo(constraints + public inputs) + 3` G1 points),
- derives the Lagrange form via an inverse FFT over the G1 points
  (no knowledge of tau needed),
- fails with the required point count if the transcript is too small.

Keys derived from a ceremony SRS are cached with a `ceremony_` prefix so they
never mix with dev keys.

### Where to get a ceremony transcript

Public BN254 ceremonies with enough points:

- **Perpetual Powers of Tau** (BN254, up to 2^28) — community conversion tools
  can re-serialize a `.ptau` file into gnark's `kzg.SRS` format.
- **Aztec Ignition** (BN254, ~100M points).

Sizing guide for ZKLR circuits (label mode, 4 features):

| Batch size | Constraints (approx) | G1 points needed |
|-----------:|---------------------:|-----------------:|
| 20  | ~170K | 2^18 + 3 |
| 40  | ~230K | 2^18 + 3 |
| 80  | ~350K | 2^19 + 3 |

(Exact counts are printed by the pipeline at setup; the loader also tells you
the required size if the file is too small.)

## Paper note

All performance numbers may be measured in dev mode (the SRS source does not
change proving/verification cost), but any deployed verification key — and the
keys behind reported *gas* measurements — must come from the ceremony path.
