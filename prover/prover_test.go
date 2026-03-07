// prover_test.go — Tests for setup, prediction, and batch pipeline.
package prover

import (
	"math"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

const (
	testW1 = -3.3144888057
	testW2 = 0.3877478051
	testB  = 281.2868128456
)

// ─── Test: Full Setup Pipeline ───────────────────────────────

func TestSetupFull(t *testing.T) {
	result, err := RunSetup()
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	t.Logf("Constraints: %d, Compile: %v, Setup: %v", result.NumConstraints, result.CompileTime, result.SetupTime)
	t.Logf("PK: %.1f KB, VK: %.1f KB", float64(result.PKSizeBytes)/1024, float64(result.VKSizeBytes)/1024)

	// Verify with a proof
	assignment := ComputeWitness(testW1, testW2, testB, 170, 70)
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, err := plonk.Prove(result.ConstraintSystem, result.ProvingKey, witness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	publicWitness, _ := witness.Public()
	if err := plonk.Verify(proof, result.VerificationKey, publicWitness); err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	t.Log("Full pipeline: compile → setup → prove → verify ✓")
}

// ─── Test: Key Serialization ─────────────────────────────────

func TestKeySerialization(t *testing.T) {
	result, err := RunSetup()
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	tmpDir := t.TempDir()
	pkPath := filepath.Join(tmpDir, "proving.key")
	vkPath := filepath.Join(tmpDir, "verification.key")

	if err := SaveProvingKey(result.ProvingKey, pkPath); err != nil {
		t.Fatalf("Save PK: %v", err)
	}
	if err := SaveVerificationKey(result.VerificationKey, vkPath); err != nil {
		t.Fatalf("Save VK: %v", err)
	}

	pkInfo, _ := os.Stat(pkPath)
	vkInfo, _ := os.Stat(vkPath)
	t.Logf("PK file: %.1f KB, VK file: %.1f KB", float64(pkInfo.Size())/1024, float64(vkInfo.Size())/1024)

	pkLoaded, err := LoadProvingKey(pkPath)
	if err != nil {
		t.Fatalf("Load PK: %v", err)
	}
	vkLoaded, err := LoadVerificationKey(vkPath)
	if err != nil {
		t.Fatalf("Load VK: %v", err)
	}

	// Prove with loaded keys
	assignment := ComputeWitness(testW1, testW2, testB, 160, 60)
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, err := plonk.Prove(result.ConstraintSystem, pkLoaded, witness)
	if err != nil {
		t.Fatalf("Prove with loaded PK: %v", err)
	}
	publicWitness, _ := witness.Public()
	if err := plonk.Verify(proof, vkLoaded, publicWitness); err != nil {
		t.Fatalf("Verify with loaded VK: %v", err)
	}
	t.Log("Key round-trip: save → load → prove → verify ✓")
}

// ─── Test: ComputeWitness ────────────────────────────────────

func TestComputeWitness(t *testing.T) {
	tests := []struct {
		height   int
		weight   int // weight×10 (e.g., 500 = 50.0 kg)
		wantPass bool
	}{
		{180, 500, false},  // 50.0kg, BMI 15.4 -> NORMAL
		{170, 700, false},  // 70.0kg, BMI 24.2 -> NORMAL
		{160, 800, true},   // 80.0kg, BMI 31.2 -> OVERWEIGHT
		{190, 600, false},  // 60.0kg, BMI 16.6 -> NORMAL
	}

	for _, tc := range tests {
		assignment := ComputeWitness(testW1, testW2, testB, tc.height, tc.weight)
		prob := GetProbability(assignment)
		pred := GetPrediction(assignment)
		isPass := pred == "OVERWEIGHT"

		if isPass != tc.wantPass {
			t.Errorf("h=%d, w=%d: got %s (%.4f), want pass=%v", tc.height, tc.weight, pred, prob, tc.wantPass)
		} else {
			t.Logf("h=%d, w=%d: prob=%.4f → %s ✓", tc.height, tc.weight, prob, pred)
		}
	}
}

// ─── Test: Predict Pipeline ──────────────────────────────────

func TestPredict(t *testing.T) {
	setup, err := RunSetup()
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	result, err := Predict(setup, testW1, testW2, testB, 160, 850)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if !result.Verified {
		t.Fatal("Proof not verified")
	}
	if result.Prediction != "OVERWEIGHT" {
		t.Errorf("Expected OVERWEIGHT, got %s", result.Prediction)
	}
	t.Logf("h=160, w=850: prob=%.4f, pred=%s, prove=%v, verify=%v ✓",
		result.Probability, result.Prediction, result.ProveTime, result.VerifyTime)
}

// ─── Test: Verifier Only ─────────────────────────────────────

func TestVerifierOnly(t *testing.T) {
	setup, err := RunSetup()
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	assignment := ComputeWitness(testW1, testW2, testB, 175, 750)
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	proof, err := plonk.Prove(setup.ConstraintSystem, setup.ProvingKey, witness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}

	// Verifier only has VK + public inputs
	publicWitness, _ := witness.Public()
	if err := plonk.Verify(proof, setup.VerificationKey, publicWitness); err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	t.Log("Verifier used ONLY VK + public inputs (X, Y) ✓")
}

// ─── Test: Sigmoid Table (moved from circuit_test) ───────────

func TestSigmoidTable(t *testing.T) {
	table := circuit.BuildSigmoidTable()

	expectedSize := circuit.SigmoidRange*(1<<circuit.InputPrecision) + 1
	if len(table) != expectedSize {
		t.Fatalf("Expected %d entries, got %d", expectedSize, len(table))
	}

	offset := circuit.SigmoidOffset * (1 << circuit.InputPrecision)

	checks := []struct {
		index    int
		expected float64
		name     string
	}{
		{offset, 0.5, "sigmoid(0)"},
		{offset + 1024, 1.0 / (1.0 + math.Exp(-1.0)), "sigmoid(1)"},
		{offset + 2048, 1.0 / (1.0 + math.Exp(-2.0)), "sigmoid(2)"},
		{offset + 4096, 1.0 / (1.0 + math.Exp(-4.0)), "sigmoid(4)"},
		{offset - 4096, 1.0 / (1.0 + math.Exp(4.0)), "sigmoid(-4)"},
		{0, circuit.SigmoidFloat(-float64(circuit.SigmoidOffset)), "sigmoid(min)"},
	}

	for _, c := range checks {
		actual := float64(table[c.index].Int64()) / float64(1<<circuit.OutputPrecision)
		diff := math.Abs(actual - c.expected)
		if diff > 0.001 {
			t.Errorf("%s: expected %.4f, got %.4f (diff=%.6f)", c.name, c.expected, actual, diff)
		} else {
			t.Logf("%s: %.4f ✓ (diff=%.6f)", c.name, actual, diff)
		}
	}
}

// ─── Test: Batch Predict ─────────────────────────────────────

func TestBatchPredict(t *testing.T) {
	setup, err := RunBatchSetup(5)
	if err != nil {
		t.Fatalf("Batch setup failed: %v", err)
	}
	t.Logf("Batch circuit: %d constraints (batch=5)", setup.NumConstraints)

	features := [][2]int{{150, 400}, {160, 500}, {170, 700}, {180, 900}, {190, 1000}}
	results := BatchPredictParallel(setup, testW1, testW2, testB, features, 2)

	for _, br := range results {
		if br.Error != nil {
			t.Errorf("Batch %d failed: %v", br.BatchIndex, br.Error)
			continue
		}
		if !br.Verified {
			t.Errorf("Batch %d not verified", br.BatchIndex)
		}
		for _, p := range br.Predictions {
			t.Logf("  h=%d, w=%d: prob=%.4f → %s", p.Height, p.Weight, p.Probability, p.Prediction)
		}
		t.Logf("Batch %d: prove=%v, verify=%v ✓", br.BatchIndex, br.ProveTime, br.VerifyTime)
	}
}

// ─── Test: Compare One Input vs Two Inputs ───────────────────

// TestCompareOneVsTwoInputs compares predictions made with a single feature
// (height only, W2=0) against predictions made with two features (height +
// weight). It logs a side-by-side table so the impact of the second feature
// can be inspected.
func TestCompareOneVsTwoInputs(t *testing.T) {
	samples := [][2]int{
		{150, 400},
		{160, 600},
		{165, 700},
		{170, 750},
		{175, 850},
		{180, 950},
		{185, 1000},
		{190, 600},
	}

	t.Log("─────────────────────────────────────────────────────────────────────")
	t.Log("h    w     | one-input prob  pred      | two-input prob  pred")
	t.Log("─────────────────────────────────────────────────────────────────────")

	for _, s := range samples {
		h, w := s[0], s[1]

		// One-input model: only height feature, W2 = 0
		oneInput := ComputeWitness(testW1, 0, testB, h, w)
		oneProb := GetProbability(oneInput)
		onePred := GetPrediction(oneInput)

		// Two-input model: height + weight features
		twoInput := ComputeWitness(testW1, testW2, testB, h, w)
		twoProb := GetProbability(twoInput)
		twoPred := GetPrediction(twoInput)

		t.Logf("h=%3d w=%4d | one: %.4f %-10s | two: %.4f %-10s",
			h, w, oneProb, onePred, twoProb, twoPred)
	}
	t.Log("─────────────────────────────────────────────────────────────────────")
}

// ─── Test: Multiple Marks ────────────────────────────────────

func TestMultipleFeatures(t *testing.T) {
	setup, err := RunSetup()
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	featuresList := [][2]int{{150, 400}, {160, 500}, {170, 600}, {170, 750}, {180, 800}, {185, 950}}

	for _, f := range featuresList {
		assignment := ComputeWitness(testW1, testW2, testB, f[0], f[1])
		witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
		if err != nil {
			t.Errorf("h=%d, w=%d: witness failed: %v", f[0], f[1], err)
			continue
		}

		proof, err := plonk.Prove(setup.ConstraintSystem, setup.ProvingKey, witness)
		if err != nil {
			t.Errorf("h=%d, w=%d: prove failed: %v", f[0], f[1], err)
			continue
		}

		publicWitness, _ := witness.Public()
		err = plonk.Verify(proof, setup.VerificationKey, publicWitness)
		if err != nil {
			t.Errorf("h=%d, w=%d: verify failed: %v", f[0], f[1], err)
		} else {
			prob := GetProbability(assignment)
			pred := GetPrediction(assignment)
			yBig := assignment.Y.(*big.Int)
			zt := assignment.ZTable.(*big.Int)
			t.Logf("h=%3d, w=%3d: z_table=%-6v y=%-6v prob=%.4f → %s ✓", f[0], f[1], zt, yBig, prob, pred)
		}
	}
}
