// srs.go — Loading a real (ceremony) KZG SRS.
//
// unsafekzg generates an SRS from a locally known tau — fine for development
// and benchmarking, but anyone who knows tau can forge proofs, so keys derived
// from it must never back a production verifier or an on-chain contract.
//
// For production, load a gnark-serialized BN254 SRS derived from a public
// ceremony (e.g. Aztec Ignition or Perpetual Powers of Tau, converted with a
// community tool) via -srs=<file>. See docs/TRUSTED_SETUP.md.
package prover

import (
	"fmt"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	bn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	kzgbn254 "github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	kzggeneric "github.com/consensys/gnark-crypto/kzg"
	"github.com/consensys/gnark/constraint"
)

// LoadCeremonySRS reads a gnark-serialized BN254 KZG SRS from disk, truncates
// it to the size the constraint system needs, and derives the Lagrange form
// (an inverse FFT over the G1 points — no knowledge of tau required).
// Returns (canonical, lagrange) ready for plonk.Setup.
func LoadCeremonySRS(path string, ccs constraint.ConstraintSystem) (kzggeneric.SRS, kzggeneric.SRS, error) {
	// Same sizing rule as gnark's plonk.Setup expects (see unsafekzg.NewSRS):
	sizeSystem := ccs.GetNbConstraints() + ccs.GetNbPublicVariables()
	sizeLagrange := ecc.NextPowerOfTwo(uint64(sizeSystem))
	sizeCanonical := sizeLagrange + 3

	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open SRS file: %w", err)
	}
	defer f.Close()

	var full kzgbn254.SRS
	if _, err := full.ReadFrom(f); err != nil {
		return nil, nil, fmt.Errorf("read SRS file %q: %w", path, err)
	}

	if uint64(len(full.Pk.G1)) < sizeCanonical {
		return nil, nil, fmt.Errorf(
			"SRS file %q has %d G1 points; circuit needs %d (constraints=%d). Use a larger ceremony transcript",
			path, len(full.Pk.G1), sizeCanonical, ccs.GetNbConstraints(),
		)
	}

	canonical := &kzgbn254.SRS{Vk: full.Vk}
	canonical.Pk.G1 = full.Pk.G1[:sizeCanonical]

	// Lagrange form over the first 2^k points (ToLagrangeG1 requires a power
	// of two). Copy first: ToLagrangeG1 transforms in place.
	coeffs := make([]bn254.G1Affine, sizeLagrange)
	copy(coeffs, canonical.Pk.G1[:sizeLagrange])
	lagrangeG1, err := kzgbn254.ToLagrangeG1(coeffs)
	if err != nil {
		return nil, nil, fmt.Errorf("convert SRS to Lagrange form: %w", err)
	}

	lagrange := &kzgbn254.SRS{Vk: full.Vk}
	lagrange.Pk.G1 = lagrangeG1

	return canonical, lagrange, nil
}
