package main

import (
	"flag"
	"log"
	"time"

	"github.com/santhoshcheemala/ZKLR/simulation"
)

func main() {
	animated := flag.Bool("animated", false, "Run animated simulation (fast, no real proofs)")
	latency := flag.Duration("latency", 100*time.Millisecond, "Simulated network latency")
	flag.Parse()

	datasetFile := "data/student_dataset_test.csv"
	modelFile := "data/best_model_parameters.txt"

	if *animated {
		sim, err := simulation.NewNetworkSimulation(datasetFile, modelFile, *latency)
		if err != nil {
			log.Fatalf("Failed to create simulation: %v", err)
		}

		if err := sim.RunDistributed(); err != nil {
			log.Fatalf("Simulation failed: %v", err)
		}
	} else {
		log.Println("Starting actual proof generation...")
		log.Println("This will take several minutes. Use -animated flag for quick simulation.")
		simulation.RunWithActualProofs()
	}
}
