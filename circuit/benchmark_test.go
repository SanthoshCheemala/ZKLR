// benchmark_test.go — Performance benchmarks for the LR prediction circuit.
//
// Measures the key metrics for circuit efficiency:
//   - Constraint count (circuit size)
//   - Compile time (SCS generation)
//   - Setup time (SRS + key generation)
//   - Prove time
//   - Verify time
//   - Proof size (bytes)
//
// Run with:   go test -bench=. -benchmem ./circuit/ -count=1 -timeout=300s
// Save to:    go test -bench=. -benchmem ./circuit/ -count=1 -timeout=300s > results/benchmark.txt
package circuit

import (
	"bytes"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test"
	"github.com/consensys/gnark/test/unsafekzg"
)

// ─── Benchmark: Full Pipeline ────────────────────────────────
// Measures compile → setup → prove → verify as one pipeline.

func BenchmarkFullPipeline(b *testing.B) {
	// Prepare a valid witness
	w, bi, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)

	for i := 0; i < b.N; i++ {
		// Compile
		circuit := &LRCircuit{}
		ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
		if err != nil {
			b.Fatal(err)
		}

		// Setup
		srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
		if err != nil {
			b.Fatal(err)
		}
		pk, vk, err := plonk.Setup(ccs, srs, srsLagrange)
		if err != nil {
			b.Fatal(err)
		}

		// Prove
		assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: bi, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
		witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
		proof, err := plonk.Prove(ccs, pk, witness)
		if err != nil {
			b.Fatal(err)
		}

		// Verify
		publicWitness, _ := witness.Public()
		err = plonk.Verify(proof, vk, publicWitness)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// ─── Benchmark: Individual Phases ────────────────────────────

func BenchmarkCompile(b *testing.B) {
	for i := 0; i < b.N; i++ {
		circuit := &LRCircuit{}
		_, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSetup(b *testing.B) {
	circuit := &LRCircuit{}
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srs, srsLagrange, _ := unsafekzg.NewSRS(ccs)
		_, _, err := plonk.Setup(ccs, srs, srsLagrange)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProve(b *testing.B) {
	circuit := &LRCircuit{}
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
	srs, srsLagrange, _ := unsafekzg.NewSRS(ccs)
	pk, _, _ := plonk.Setup(ccs, srs, srsLagrange)

	w, bi, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)
	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: bi, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := plonk.Prove(ccs, pk, witness)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVerify(b *testing.B) {
	circuit := &LRCircuit{}
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
	srs, srsLagrange, _ := unsafekzg.NewSRS(ccs)
	pk, vk, _ := plonk.Setup(ccs, srs, srsLagrange)

	w, bi, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)
	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: bi, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, _ := plonk.Prove(ccs, pk, witness)
	publicWitness, _ := witness.Public()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := plonk.Verify(proof, vk, publicWitness)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// ─── Detailed Metrics Report ─────────────────────────────────
// Not a benchmark per se — runs once and prints a detailed metrics report.
// Run with: go test -run=TestMetricsReport ./circuit/ -v -timeout=300s

func TestMetricsReport(t *testing.T) {
	w, bi, x, zt, rem, y := computeWitness(-3.3144933046, 0.3877500778, 281.2861173099, 170, 750)

	// 1. Compile
	t.Log("─── Circuit Compilation ───")
	startCompile := time.Now()
	circuit := &LRCircuit{}
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
	compileTime := time.Since(startCompile)
	if err != nil {
		t.Fatal(err)
	}

	r1cs := ccs.(*cs.SparseR1CS)
	nbConstraints := ccs.GetNbConstraints()
	nbVariables := r1cs.GetNbInternalVariables() + r1cs.GetNbPublicVariables() + r1cs.GetNbSecretVariables()

	t.Logf("  Constraints:      %d", nbConstraints)
	t.Logf("  Variables:        %d", nbVariables)
	t.Logf("  Public inputs:    %d", r1cs.GetNbPublicVariables())
	t.Logf("  Secret inputs:    %d", r1cs.GetNbSecretVariables())
	t.Logf("  Internal vars:    %d", r1cs.GetNbInternalVariables())
	t.Logf("  Compile time:     %v", compileTime)

	// 2. Setup
	t.Log("\n─── Trusted Setup (SRS + Keys) ───")
	startSetup := time.Now()
	srs, srsLagrange, _ := unsafekzg.NewSRS(ccs)
	pk, vk, err := plonk.Setup(ccs, srs, srsLagrange)
	setupTime := time.Since(startSetup)
	if err != nil {
		t.Fatal(err)
	}

	var pkBuf, vkBuf bytes.Buffer
	pk.WriteTo(&pkBuf)
	vk.WriteTo(&vkBuf)
	t.Logf("  Setup time:       %v", setupTime)
	t.Logf("  Proving key size: %d bytes (%.1f KB)", pkBuf.Len(), float64(pkBuf.Len())/1024)
	t.Logf("  Verif. key size:  %d bytes (%.1f KB)", vkBuf.Len(), float64(vkBuf.Len())/1024)

	// 3. Prove
	t.Log("\n─── Proof Generation ───")
	assignment := &LRCircuit{W: [2]frontend.Variable{w[0], w[1]}, B: bi, X: [2]frontend.Variable{x[0], x[1]}, ZTable: zt, Rem: rem, Y: y}
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())

	startProve := time.Now()
	proof, err := plonk.Prove(ccs, pk, witness)
	proveTime := time.Since(startProve)
	if err != nil {
		t.Fatal(err)
	}

	var proofBuf bytes.Buffer
	proof.WriteTo(&proofBuf)
	t.Logf("  Prove time:       %v", proveTime)
	t.Logf("  Proof size:       %d bytes", proofBuf.Len())

	// 4. Verify
	t.Log("\n─── Proof Verification ───")
	publicWitness, _ := witness.Public()

	startVerify := time.Now()
	err = plonk.Verify(proof, vk, publicWitness)
	verifyTime := time.Since(startVerify)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("  Verify time:      %v", verifyTime)
	t.Logf("  Verified:         ✓")

	// 5. Summary report
	t.Log("\n═══════════════════════════════════════")
	t.Log("     CIRCUIT METRICS SUMMARY")
	t.Log("═══════════════════════════════════════")
	t.Logf("  Constraints:      %d", nbConstraints)
	t.Logf("  Variables:        %d", nbVariables)
	t.Logf("  Compile:          %v", compileTime)
	t.Logf("  Setup:            %v", setupTime)
	t.Logf("  Prove:            %v", proveTime)
	t.Logf("  Verify:           %v", verifyTime)
	t.Logf("  Proof size:       %d bytes", proofBuf.Len())
	t.Logf("  PK size:          %.1f KB", float64(pkBuf.Len())/1024)
	t.Logf("  VK size:          %.1f KB", float64(vkBuf.Len())/1024)
	t.Log("═══════════════════════════════════════")

	// 6. Save to results file
	saveMetricsReport(t, nbConstraints, nbVariables, compileTime, setupTime,
		proveTime, verifyTime, proofBuf.Len(), pkBuf.Len(), vkBuf.Len())
}

func saveMetricsReport(t *testing.T, constraints, variables int,
	compile, setup, prove, verify time.Duration,
	proofSize, pkSize, vkSize int) {

	resultsDir := filepath.Join("..", "results")
	os.MkdirAll(resultsDir, 0o755)

	path := filepath.Join(resultsDir, "circuit_metrics.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Logf("Warning: could not save metrics: %v", err)
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "ZK-LR Circuit Metrics Report\n")
	fmt.Fprintf(f, "Generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(f, "Curve: BN254\n")
	fmt.Fprintf(f, "Backend: PLONK\n\n")
	fmt.Fprintf(f, "Constraints:      %d\n", constraints)
	fmt.Fprintf(f, "Variables:        %d\n", variables)
	fmt.Fprintf(f, "Compile time:     %v\n", compile)
	fmt.Fprintf(f, "Setup time:       %v\n", setup)
	fmt.Fprintf(f, "Prove time:       %v\n", prove)
	fmt.Fprintf(f, "Verify time:      %v\n", verify)
	fmt.Fprintf(f, "Proof size:       %d bytes\n", proofSize)
	fmt.Fprintf(f, "Proving key:      %d bytes (%.1f KB)\n", pkSize, float64(pkSize)/1024)
	fmt.Fprintf(f, "Verification key: %d bytes (%.1f KB)\n", vkSize, float64(vkSize)/1024)

	// Ratios
	fmt.Fprintf(f, "\nEfficiency Ratios:\n")
	fmt.Fprintf(f, "  Prove/Verify ratio:  %.0fx\n", prove.Seconds()/verify.Seconds())
	fmt.Fprintf(f, "  Prove/constraint:    %.2f µs\n", prove.Seconds()*1e6/float64(constraints))

	t.Logf("Metrics saved → %s", path)
}

// ─── Test: Witness for pre-scaled integer constants ──────────
// Uses the exact W_SCALED and B_SCALED from model_weights.txt

func TestWithPreScaledConstants(t *testing.T) {
	// W1_SCALED = -14235640345
	// W2_SCALED = 1665373903
	// B_SCALED  = 1208114674664
	w1Scaled := new(big.Int).SetInt64(-14235640345)
	w2Scaled := new(big.Int).SetInt64(1665373903)
	bScaled := new(big.Int).SetInt64(1208114674664)

	dataToTest := [][2]int{
		{150, 500}, {160, 600}, {170, 750}, {180, 850}, {190, 1000},
	}

	for _, features := range dataToTest {
		h, w := features[0], features[1]
		x_bi := [2]*big.Int{big.NewInt(int64(h)), big.NewInt(int64(w))}

		// z_linear = W1*H + W2*W + B
		w1x1 := new(big.Int).Mul(w1Scaled, x_bi[0])
		w2x2 := new(big.Int).Mul(w2Scaled, x_bi[1])
		z_linear := new(big.Int).Add(w1x1, w2x2)
		z_linear.Add(z_linear, bScaled)

		// z_shifted = z_linear + offset
		offsetScaled := new(big.Int).Lsh(big.NewInt(ModelOffset), Precision)
		z_shifted := new(big.Int).Add(z_linear, offsetScaled)

		// z_table = z_shifted / 2^22
		shiftFactor := new(big.Int).Lsh(big.NewInt(1), 22) // Precision-InputPrecision is 22
		z_table := new(big.Int).Div(z_shifted, shiftFactor)
		rem := new(big.Int).Rem(z_shifted, shiftFactor)

		// Clamp z_table 
		LowerBound := int64((ModelOffset - SigmoidOffset) * (1 << InputPrecision))
		UpperBound := int64((ModelOffset + SigmoidOffset) * (1 << InputPrecision))

		z_table_clamped := new(big.Int).Set(z_table)
		if z_table_clamped.Int64() < LowerBound {
			z_table_clamped = big.NewInt(LowerBound)
		} else if z_table_clamped.Int64() > UpperBound {
			z_table_clamped = big.NewInt(UpperBound)
		}

		z_index := z_table_clamped.Int64() - LowerBound

		// sigmoid
		z_f := float64(z_index) / float64(1<<InputPrecision) - float64(SigmoidOffset)
		y_float := SigmoidFloat(z_f)
		y_bi := new(big.Int).SetInt64(int64(y_float * float64(1<<OutputPrecision)))
		maxOut := big.NewInt((1 << OutputPrecision) - 1)
		if y_bi.Cmp(maxOut) > 0 {
			y_bi = maxOut
		}

		assignment := &LRCircuit{W: [2]frontend.Variable{w1Scaled, w2Scaled}, B: bScaled, X: [2]frontend.Variable{x_bi[0], x_bi[1]}, ZTable: z_table, Rem: rem, Y: y_bi}
		err := test.IsSolved(&LRCircuit{}, assignment, ecc.BN254.ScalarField())

		prediction := "NORMAL"
		if y_float >= 0.5 {
			prediction = "OVERWEIGHT"
		}

		if err != nil {
			t.Errorf("h=%d, w=%d: REJECTED (%v)", h, w, err)
		} else {
			t.Logf("h=%3d, w=%3d: z_table=%v y=%.4f → %s ✓", h, w, z_table, y_float, prediction)
		}
	}
}
