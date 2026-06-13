// cmd/verify — Independent proof verification.
//
// Verifies PLONK proofs from files alone: verification key + proof + public
// witness (features, outputs, model commitment). It shares no process state
// with the prover — this is the artifact that demonstrates a third party can
// check predictions without the proving key, the weights, or the dataset.
//
// Usage:
//
//	go run ./cmd/verify -dir=results/proofs                  # verify all batches in a directory
//	go run ./cmd/verify -vk=vk.key -proof=batch_000.proof -public=batch_000.pubw
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"

	"github.com/santhoshcheemala/ZKLR/prover"
)

func main() {
	dir := flag.String("dir", "", "directory of exported artifacts (vk.key, batch_NNN.proof, batch_NNN.pubw)")
	vkPath := flag.String("vk", "", "verification key file (single-proof mode)")
	proofPath := flag.String("proof", "", "proof file (single-proof mode)")
	publicPath := flag.String("public", "", "public witness file (single-proof mode)")
	flag.Parse()

	switch {
	case *dir != "":
		verifyDir(*dir)
	case *vkPath != "" && *proofPath != "" && *publicPath != "":
		vk, err := prover.LoadVerificationKey(*vkPath)
		if err != nil {
			fail("load vk: %v", err)
		}
		elapsed, err := verifyOne(vk, *proofPath, *publicPath)
		if err != nil {
			fail("INVALID: %v", err)
		}
		fmt.Printf("VALID  %s  (%v)\n", filepath.Base(*proofPath), elapsed.Round(time.Microsecond))
	default:
		fmt.Fprintln(os.Stderr, "Usage: verify -dir=<dir>  |  verify -vk=<f> -proof=<f> -public=<f>")
		os.Exit(2)
	}
}

func verifyDir(dir string) {
	vk, err := prover.LoadVerificationKey(filepath.Join(dir, "vk.key"))
	if err != nil {
		fail("load vk: %v", err)
	}

	if manifest, err := os.ReadFile(filepath.Join(dir, "manifest.txt")); err == nil {
		fmt.Println("Manifest:")
		for _, line := range strings.Split(strings.TrimSpace(string(manifest)), "\n") {
			fmt.Println("  " + line)
		}
	}

	proofs, err := filepath.Glob(filepath.Join(dir, "batch_*.proof"))
	if err != nil || len(proofs) == 0 {
		fail("no batch_*.proof files found in %s", dir)
	}
	sort.Strings(proofs)

	valid, invalid := 0, 0
	var total time.Duration
	for _, p := range proofs {
		pubw := strings.TrimSuffix(p, ".proof") + ".pubw"
		elapsed, err := verifyOne(vk, p, pubw)
		total += elapsed
		if err != nil {
			fmt.Printf("INVALID  %s: %v\n", filepath.Base(p), err)
			invalid++
			continue
		}
		fmt.Printf("VALID    %s  (%v)\n", filepath.Base(p), elapsed.Round(time.Microsecond))
		valid++
	}

	fmt.Printf("\n%d valid, %d invalid — total verify time %v (avg %v/batch)\n",
		valid, invalid, total.Round(time.Millisecond), (total / time.Duration(len(proofs))).Round(time.Microsecond))
	if invalid > 0 {
		os.Exit(1)
	}
}

// verifyOne checks one proof against one public witness and returns the
// verification time.
func verifyOne(vk plonk.VerifyingKey, proofPath, publicPath string) (time.Duration, error) {
	proof := plonk.NewProof(ecc.BN254)
	if err := readInto(proofPath, proof.(io.ReaderFrom)); err != nil {
		return 0, fmt.Errorf("load proof: %w", err)
	}

	pubWitness, err := witness.New(ecc.BN254.ScalarField())
	if err != nil {
		return 0, fmt.Errorf("new witness: %w", err)
	}
	if err := readInto(publicPath, pubWitness); err != nil {
		return 0, fmt.Errorf("load public witness: %w", err)
	}

	start := time.Now()
	err = plonk.Verify(proof, vk, pubWitness)
	return time.Since(start), err
}

func readInto(path string, dst io.ReaderFrom) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = dst.ReadFrom(f)
	return err
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
