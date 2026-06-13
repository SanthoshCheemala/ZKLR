// cmd/constraint_scan — Empirical input for the amortization cost model (F1).
//
// Compiles the BatchLabelCircuit for a series of batch sizes (compile only,
// no setup) and emits batch,constraints as CSV. The fit T(B) = a + b·B over
// these points is the constraint half of the cost model; the prove-time half
// comes from benchmark runs.
//
// Usage:
//
//	go run ./cmd/constraint_scan -features=9 -batches=1,5,10,20,40,80 \
//	    -out=results/constraint_scaling.csv
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"

	"github.com/santhoshcheemala/ZKLR/circuit"
)

func main() {
	featuresFlag := flag.Int("features", 4, "number of model features")
	batchesFlag := flag.String("batches", "1,5,10,20,40,80", "comma-separated batch sizes")
	modeFlag := flag.String("mode", "label", "circuit mode: label, prob, or ovr (one-vs-rest)")
	classesFlag := flag.Int("classes", 3, "number of classes (ovr mode only)")
	outPath := flag.String("out", "", "CSV output path (default stdout only)")
	flag.Parse()

	var batches []int
	for _, s := range strings.Split(*batchesFlag, ",") {
		b, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || b < 1 {
			fmt.Fprintf(os.Stderr, "invalid batch size %q\n", s)
			os.Exit(1)
		}
		batches = append(batches, b)
	}

	classes := 1
	if *modeFlag == "ovr" {
		classes = *classesFlag
	}

	var sb strings.Builder
	sb.WriteString("batch,classes,features,mode,units,constraints,compile_ms\n")

	fmt.Printf("%8s %8s %10s %8s %12s %12s\n", "batch", "classes", "features", "units", "constraints", "compile")
	for _, b := range batches {
		var c frontend.Circuit
		switch *modeFlag {
		case "label":
			c = circuit.NewBatchLabelCircuit(b, *featuresFlag)
		case "prob":
			c = circuit.NewBatchCircuit(b, *featuresFlag)
		case "ovr":
			c = circuit.NewOneVsRestBatchCircuit(b, classes, *featuresFlag)
		default:
			fmt.Fprintf(os.Stderr, "invalid mode %q\n", *modeFlag)
			os.Exit(1)
		}

		start := time.Now()
		ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, c)
		if err != nil {
			fmt.Fprintf(os.Stderr, "compile batch=%d: %v\n", b, err)
			os.Exit(1)
		}
		elapsed := time.Since(start)

		n := ccs.GetNbConstraints()
		units := b * classes // amortization units: sigmoid evaluations sharing the table
		fmt.Printf("%8d %8d %10d %8d %12d %12v\n", b, classes, *featuresFlag, units, n, elapsed.Round(time.Millisecond))
		fmt.Fprintf(&sb, "%d,%d,%d,%s,%d,%d,%d\n", b, classes, *featuresFlag, *modeFlag, units, n, elapsed.Milliseconds())
	}

	if *outPath != "" {
		if err := os.WriteFile(*outPath, []byte(sb.String()), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", *outPath, err)
			os.Exit(1)
		}
		fmt.Printf("\nWrote %s\n", *outPath)
	}
}
