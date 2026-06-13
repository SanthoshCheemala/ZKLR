# ZKLR — build, test, and reproduce every reported number.

GO ?= go
PY ?= python3

.PHONY: build test reproduce datasets cost-model baselines gas clean help

help:
	@echo "Targets:"
	@echo "  build       go build ./..."
	@echo "  test        full test suite incl. tamper matrix (~90s)"
	@echo "  reproduce   datasets + fidelity + cost model + Groth16 + EZKL baselines"
	@echo "  gas         on-chain gas benchmark (requires Foundry)"
	@echo "  clean       remove generated keys/artifacts (results stay)"

build:
	$(GO) build ./...

test: build
	$(GO) test ./... -count=1

# ─── Reproduce the paper numbers ─────────────────────────────

WBC_W = 0.5198540959100377,0.29788720690316345,0.31861482728846857,0.29596380922786863,0.19057230570348138,0.6431713625628626,0.1292808531285863,0.1883990254615792,0.7134676765886211
WBC_B = -11.069237857799271

reproduce: datasets cost-model baselines
	@echo ""
	@echo "Reproduced: fidelity (results/fidelity_*.txt, results/fig_fidelity_*.png),"
	@echo "cost model (results/cost_model.txt, results/fig_cost_model.png),"
	@echo "baselines (results/groth16_baseline_wbc_b20.txt, results/ezkl_baseline_b20.txt)."
	@echo "On-chain gas: make gas (requires Foundry). HPC sweep: scripts/run_hpc_sweep.sh."

datasets:
	$(PY) scripts/prepare_real_datasets.py

cost-model: build
	$(GO) run ./cmd/constraint_scan -features=9 -batches=1,5,10,20,40,80 -out=results/constraint_scaling_f9.csv
	$(GO) run ./cmd/constraint_scan -mode=ovr -classes=3 -features=4 -batches=1,5,10,20,40 -out=results/constraint_scaling_ovr_c3_f4.csv
	$(PY) scripts/fit_cost_model.py --constraints=results/constraint_scaling_f9.csv

baselines: build datasets
	$(GO) run ./cmd/groth16_baseline -batch=20 -runs=3 -dataset=data/wbc_test.csv \
		-weights="$(WBC_W)" -bias=$(WBC_B) | tee results/groth16_baseline_wbc_b20.txt
	$(PY) scripts/ezkl_baseline.py --batch 20

gas:
	./scripts/run_gas_benchmark.sh 20 data/wbc_test.csv "$(WBC_W)" $(WBC_B)

clean:
	rm -f results/*.key
	rm -rf results/proofs_* results/ezkl_baseline contracts/out contracts/cache
