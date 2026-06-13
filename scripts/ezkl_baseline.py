#!/usr/bin/env python3
"""EZKL baseline for the head-to-head comparison (T6).

Proves the SAME statement as ZKLR's batch circuit — "the (private) LR model
predicts these outputs for these (public) inputs" — using EZKL (halo2/KZG)
on the same WBC model and a batch of test samples, and reports timings,
proof size and key sizes.

The model is built as a minimal ONNX graph (MatMul + Add + Sigmoid) with the
exact weights ZKLR uses, so both systems prove the same function. Visibility
matches ZKLR's prob mode: inputs public, outputs public, params private.
(EZKL has no built-in label-only mode — note this in the comparison.)

Usage:
  python3 scripts/ezkl_baseline.py --batch 20
"""

import argparse
import asyncio
import inspect
import json
import os
import time

import numpy as np
import onnx
import pandas as pd
from onnx import TensorProto, helper

WORK = "results/ezkl_baseline"


def build_onnx(weights: np.ndarray, bias: float, batch: int, path: str):
    """Minimal LR graph: sigmoid(X @ W^T + b), standard ONNX ops only."""
    f = len(weights)
    W = helper.make_tensor("W", TensorProto.FLOAT, [f, 1], weights.astype(np.float32).tobytes(), raw=True)
    Bc = helper.make_tensor("B", TensorProto.FLOAT, [1], np.array([bias], dtype=np.float32).tobytes(), raw=True)

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "B"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "zklr_lr",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, f])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 1])],
        initializer=[W, Bc],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--dataset", default="data/wbc_test.csv")
    ap.add_argument("--weights", default="data/wbc_weights.txt")
    args = ap.parse_args()

    import ezkl

    os.makedirs(WORK, exist_ok=True)
    paths = {n: os.path.join(WORK, n) for n in
             ["model.onnx", "settings.json", "model.compiled", "input.json",
              "witness.json", "vk.key", "pk.key", "proof.json"]}

    # ─── Same model + data as the ZKLR WBC run ───
    with open(args.weights) as f:
        lines = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in f if ":" in l}
    w = np.array([float(x) for x in lines["Weights"].split(",")])
    b = float(lines["Bias"])

    df = pd.read_csv(args.dataset)
    X = df.iloc[:args.batch, :-1].values.astype(np.float32)

    build_onnx(w, b, args.batch, paths["model.onnx"])
    with open(paths["input.json"], "w") as f:
        json.dump({"input_data": [X.flatten().tolist()]}, f)

    timings, sizes = {}, {}

    def step(name, fn):
        # ezkl's pyo3 bindings return awaitables for some calls and require a
        # running event loop — run every step inside one.
        async def runner():
            r = fn()
            if inspect.isawaitable(r):
                r = await r
            return r

        t0 = time.time()
        res = asyncio.run(runner())
        timings[name] = time.time() - t0
        print(f"  {name}: {timings[name]:.2f}s")
        return res

    print(f"=== EZKL baseline (batch={args.batch}, features={len(w)}) ===")
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    run_args.output_visibility = "public"
    run_args.param_visibility = "private"

    step("gen_settings", lambda: ezkl.gen_settings(paths["model.onnx"], paths["settings.json"], py_run_args=run_args))
    step("calibrate", lambda: ezkl.calibrate_settings(paths["input.json"], paths["model.onnx"], paths["settings.json"], "resources"))
    step("compile", lambda: ezkl.compile_circuit(paths["model.onnx"], paths["model.compiled"], paths["settings.json"]))
    step("get_srs", lambda: ezkl.get_srs(paths["settings.json"]))
    step("setup", lambda: ezkl.setup(paths["model.compiled"], paths["vk.key"], paths["pk.key"]))
    step("gen_witness", lambda: ezkl.gen_witness(paths["input.json"], paths["model.compiled"], paths["witness.json"]))
    step("prove", lambda: ezkl.prove(paths["witness.json"], paths["model.compiled"], paths["pk.key"], paths["proof.json"]))
    ok = step("verify", lambda: ezkl.verify(paths["proof.json"], paths["settings.json"], paths["vk.key"]))

    with open(paths["settings.json"]) as f:
        logrows = json.load(f)["run_args"]["logrows"]
    with open(paths["proof.json"]) as f:
        proof_hex = json.load(f).get("hex_proof") or ""
    sizes["proof_bytes"] = max((len(proof_hex) - 2) // 2, 0) or os.path.getsize(paths["proof.json"])
    sizes["pk_bytes"] = os.path.getsize(paths["pk.key"])
    sizes["vk_bytes"] = os.path.getsize(paths["vk.key"])

    report = os.path.join("results", f"ezkl_baseline_b{args.batch}.txt")
    with open(report, "w") as f:
        f.write(f"EZKL baseline — WBC model, batch={args.batch}, features={len(w)}\n")
        f.write(f"ezkl version: {ezkl.__version__}, logrows: {logrows}\n")
        f.write(f"verified: {ok}\n")
        for k, v in timings.items():
            f.write(f"{k:>14}: {v:.3f}s\n")
        for k, v in sizes.items():
            f.write(f"{k:>14}: {v}\n")
        f.write(f"prove/sample: {timings['prove'] / args.batch:.3f}s\n")
    print(f"\nverified: {ok}")
    print(f"proof: {sizes['proof_bytes']} bytes, pk: {sizes['pk_bytes']/1e6:.1f} MB, "
          f"logrows: {logrows}, prove/sample: {timings['prove']/args.batch:.3f}s")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
