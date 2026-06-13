#!/usr/bin/env python3
"""Aggregate an HPC sweep CSV (from run_hpc_sweep.sh) into the paper artifacts.

Inputs:
  results/hpc_sweep.csv          rows: one per (batch, workers, repetition)
  results/constraint_scaling.csv constraints vs batch (from the same sweep)

Outputs:
  results/hpc_sweep_summary.csv  median ± std of wall_per_sample per (batch,workers)
  results/fig_hpc_heatmap.png    heatmap of median wall/sample (batch × workers)
  results/fig_throughput.png     throughput vs workers per batch size
  results/cost_model_timing.txt  prove-time cost fit T_prove(B) = a + b·B
  results/fig_cost_model_timing.png

Reports median ± std so single-run noise is visible, and separates the
best configuration (the headline "optimal" point).
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/hpc_sweep.csv")
    ap.add_argument("--constraints", default="results/constraint_scaling.csv")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found — run ./run_hpc_sweep.sh first")
    df = pd.read_csv(args.csv)

    # ── Median ± std per (batch, workers) ──
    g = df.groupby(["batch", "workers"])
    summary = g.agg(
        wall_median=("wall_per_sample", "median"),
        wall_std=("wall_per_sample", "std"),
        prove_median=("prove_per_sample", "median"),
        samples=("samples", "first"),
        n=("run", "count"),
    ).reset_index()
    summary["wall_std"] = summary["wall_std"].fillna(0.0)
    summary.to_csv("results/hpc_sweep_summary.csv", index=False)

    best = summary.loc[summary["wall_median"].idxmin()]
    print("=== HPC sweep summary (median wall-clock/sample, seconds) ===")
    pivot = summary.pivot(index="batch", columns="workers", values="wall_median")
    print(pivot.round(4).to_string())
    print(f"\nBest config: batch={int(best.batch)} workers={int(best.workers)} "
          f"-> {best.wall_median:.4f} s/sample "
          f"(±{best.wall_std:.4f}, n={int(best.n)})")
    print("Wrote results/hpc_sweep_summary.csv")

    # ── Prove-time cost model: prove time for ONE batch vs B ──
    # avg prove per batch = prove_total_seconds / num_batches; num_batches ≈ samples/batch
    df["prove_per_batch"] = df["prove_total_seconds"] * df["batch"] / df["samples"]
    by_b = df.groupby("batch")["prove_per_batch"].median().reset_index()
    if len(by_b) >= 2:
        B = by_b["batch"].values.astype(float)
        T = by_b["prove_per_batch"].values
        A = np.vstack([np.ones_like(B), B]).T
        (a, b), *_ = np.linalg.lstsq(A, T, rcond=None)
        pred = a + b * B
        ss_tot = ((T - T.mean()) ** 2).sum()
        r2 = 1 - ((T - pred) ** 2).sum() / ss_tot if ss_tot > 0 else 1.0
        with open("results/cost_model_timing.txt", "w") as f:
            f.write("ZKLR prove-time cost model (per batch)\n")
            f.write("=" * 50 + "\n")
            f.write(f"T_prove(B) = {a:.3f}s + {b:.4f}s * B   (R^2 = {r2:.6f})\n")
            f.write(f"  shared (table) prove cost: {a:.3f}s\n")
            f.write(f"  marginal per-sample prove: {b*1000:.1f} ms\n\n")
            f.write(f"{'batch':>6} {'prove/batch(s)':>16}\n")
            for bb, tt in zip(by_b['batch'], by_b['prove_per_batch']):
                f.write(f"{bb:>6} {tt:>16.3f}\n")
        print(f"\nT_prove(B) = {a:.3f} + {b:.4f}·B  (R²={r2:.6f}) -> results/cost_model_timing.txt")
    else:
        a = b = r2 = None

    # ── Figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Heatmap
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("workers")
        ax.set_ylabel("batch size")
        ax.set_title("Median wall-clock per sample (s) — lower is better")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white", fontsize=7)
        fig.colorbar(im, ax=ax, label="s/sample")
        fig.tight_layout()
        fig.savefig("results/fig_hpc_heatmap.png", dpi=150)
        plt.close(fig)
        print("Wrote results/fig_hpc_heatmap.png")

        # Throughput vs workers
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for bsize, grp in summary.groupby("batch"):
            grp = grp.sort_values("workers")
            thr = 1.0 / grp["wall_median"]
            ax.plot(grp["workers"], thr, marker="o", label=f"batch={bsize}")
        ax.set_xlabel("workers")
        ax.set_ylabel("throughput (samples/s)")
        ax.set_title("Throughput vs workers")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/fig_throughput.png", dpi=150)
        plt.close(fig)
        print("Wrote results/fig_throughput.png")

        # Prove-time cost fit
        if a is not None:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.scatter(by_b["batch"], by_b["prove_per_batch"], zorder=3, label="median measured")
            xs = np.linspace(0, by_b["batch"].max() * 1.05, 100)
            ax.plot(xs, a + b * xs, "--",
                    label=f"fit: {a:.2f}s + {b*1000:.1f}ms·B (R²={r2:.4f})")
            ax.set_xlabel("batch size B")
            ax.set_ylabel("prove time per batch (s)")
            ax.set_title("Prove-time amortization")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig("results/fig_cost_model_timing.png", dpi=150)
            plt.close(fig)
            print("Wrote results/fig_cost_model_timing.png")
    except ImportError:
        print("matplotlib not available — skipped figures")


if __name__ == "__main__":
    main()
