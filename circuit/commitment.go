// commitment.go — Binding the proof to a specific model.
//
// Without a commitment, the circuit statement "there exist W, B such that
// Y = sigmoid(W·X + B)" is satisfiable for almost any (X, Y) pair: the prover
// is free to pick different weights for every proof. Publishing
// C = MiMC(W_1, ..., W_n, B) as a public input pins every proof to one fixed,
// pre-announced model: the verifier (or smart contract) checks proofs against
// the C the model owner published once.
package circuit

import (
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	frmimc "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/hash/mimc"
)

// assertCommitment constrains commitment == MiMC(W..., B) inside the circuit.
// Costs a few hundred constraints — negligible next to the ~105K sigmoid table.
func assertCommitment(api frontend.API, w []frontend.Variable, b frontend.Variable, commitment frontend.Variable) error {
	h, err := mimc.NewMiMC(api)
	if err != nil {
		return err
	}
	h.Write(w...)
	h.Write(b)
	api.AssertIsEqual(commitment, h.Sum())
	return nil
}

// ComputeCommitment computes MiMC(W..., B) outside the circuit, matching
// assertCommitment. Negative weights are reduced into the BN254 scalar field
// exactly as gnark reduces negative witness values, so both sides agree.
func ComputeCommitment(wBig []*big.Int, bBig *big.Int) *big.Int {
	h := frmimc.NewMiMC()
	var e fr.Element
	for _, w := range wBig {
		e.SetBigInt(w)
		buf := e.Bytes()
		h.Write(buf[:])
	}
	e.SetBigInt(bBig)
	buf := e.Bytes()
	h.Write(buf[:])
	return new(big.Int).SetBytes(h.Sum(nil))
}

// ─── Multi-class (one-vs-rest) commitment ────────────────────
//
// Write order: W[0] fully, then W[1], ..., then all biases B[0..C-1].

// assertCommitmentMulti constrains commitment == MiMC(W[0]..., ..., B...)
// inside the circuit.
func assertCommitmentMulti(api frontend.API, w [][]frontend.Variable, b []frontend.Variable, commitment frontend.Variable) error {
	h, err := mimc.NewMiMC(api)
	if err != nil {
		return err
	}
	for _, row := range w {
		h.Write(row...)
	}
	h.Write(b...)
	api.AssertIsEqual(commitment, h.Sum())
	return nil
}

// ComputeCommitmentMulti computes the same hash outside the circuit.
func ComputeCommitmentMulti(wBig [][]*big.Int, bBig []*big.Int) *big.Int {
	h := frmimc.NewMiMC()
	var e fr.Element
	write := func(v *big.Int) {
		e.SetBigInt(v)
		buf := e.Bytes()
		h.Write(buf[:])
	}
	for _, row := range wBig {
		for _, w := range row {
			write(w)
		}
	}
	for _, b := range bBig {
		write(b)
	}
	return new(big.Int).SetBytes(h.Sum(nil))
}
