// srs_test.go — Validates the ceremony-SRS loading path end to end.
package prover

import (
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	kzgbn254 "github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

// TestLoadCeremonySRS writes a serialized SRS file, loads it back through
// LoadCeremonySRS (which derives the Lagrange form via inverse FFT, without
// knowing tau), and runs setup → prove → verify on the batch circuit.
// This is the exact code path a real ceremony transcript would take.
func TestLoadCeremonySRS(t *testing.T) {
	const batchSize, numFeatures = 2, 2

	c := circuit.NewBatchLabelCircuit(batchSize, numFeatures)
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	// Produce an SRS file. (Here generated locally — in production this file
	// comes from a public ceremony; the loader code path is identical.)
	sizeSystem := ccs.GetNbConstraints() + ccs.GetNbPublicVariables()
	sizeCanonical := ecc.NextPowerOfTwo(uint64(sizeSystem)) + 3

	tau, err := rand.Int(rand.Reader, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	srs, err := kzgbn254.NewSRS(sizeCanonical+5, tau) // a few extra points, like a real transcript
	if err != nil {
		t.Fatalf("generate srs: %v", err)
	}

	srsPath := filepath.Join(t.TempDir(), "ceremony.srs")
	f, err := os.Create(srsPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := srs.WriteTo(f); err != nil {
		t.Fatalf("write srs: %v", err)
	}
	f.Close()

	// Load + Lagrange-convert without tau
	canonical, lagrange, err := LoadCeremonySRS(srsPath, ccs)
	if err != nil {
		t.Fatalf("LoadCeremonySRS: %v", err)
	}

	pk, vk, err := plonk.Setup(ccs, canonical, lagrange)
	if err != nil {
		t.Fatalf("plonk.Setup with loaded SRS: %v", err)
	}

	// Prove + verify a real witness through the loaded-SRS keys
	features := [][]int{{160, 800}, {180, 500}}
	assignment := ComputeBatchLabelWitness([]float64{testW1, testW2}, testB, features, batchSize)
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	proof, err := plonk.Prove(ccs, pk, witness)
	if err != nil {
		t.Fatalf("prove: %v", err)
	}
	publicWitness, _ := witness.Public()
	if err := plonk.Verify(proof, vk, publicWitness); err != nil {
		t.Fatalf("verify: %v", err)
	}
	t.Log("ceremony-SRS path: load → Lagrange convert → setup → prove → verify ✓")
}

// TestLoadCeremonySRSTooSmall: a transcript with too few points must fail
// with an actionable error, not a panic downstream.
func TestLoadCeremonySRSTooSmall(t *testing.T) {
	c := circuit.NewBatchLabelCircuit(4, 2)
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	tau, _ := rand.Int(rand.Reader, ecc.BN254.ScalarField())
	srs, err := kzgbn254.NewSRS(64, tau) // far too small
	if err != nil {
		t.Fatal(err)
	}
	srsPath := filepath.Join(t.TempDir(), "small.srs")
	f, _ := os.Create(srsPath)
	srs.WriteTo(f)
	f.Close()

	if _, _, err := LoadCeremonySRS(srsPath, ccs); err == nil {
		t.Fatal("expected error for undersized SRS")
	} else {
		t.Logf("Correctly rejected undersized SRS: %v", err)
	}
}
