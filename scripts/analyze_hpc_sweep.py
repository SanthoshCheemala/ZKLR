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

    # ── Prove-time cost model on the UNCONTENDED W=1 baseline ──
    # The gnark PLONK prover is already internally multi-threaded, so prove_total
    # at high worker counts is inflated by core oversubscription. Fitting a cost
    # model on those contended runs is meaningless — use the single-prover baseline.
    base = df[df.workers == df.workers.min()].copy()
    base["prove_per_batch"] = base["prove_total_seconds"] * base["batch"] / base["samples"]
    by_b = base.groupby("batch").agg(
        prove_per_batch=("prove_per_batch", "median"),
        prove_per_sample=("prove_per_sample", "median"),
        constraints=("constraints", "first"),
    ).reset_index()
    # PLONK proves over an FFT domain padded up to the next power of two, so the
    # real cost driver is the padded domain, not the raw constraint count.
    import math
    by_b["pow2_domain"] = by_b["constraints"].apply(lambda c: 1 << math.ceil(math.log2(c)))
    by_b["domain_fill"] = 100.0 * by_b["constraints"] / by_b["pow2_domain"]

    if len(by_b) >= 2:
        B = by_b["batch"].values.astype(float)
        T = by_b["prove_per_batch"].values
        A = np.vstack([np.ones_like(B), B]).T
        (a, b), *_ = np.linalg.lstsq(A, T, rcond=None)
        pred = a + b * B
        ss_tot = ((T - T.mean()) ** 2).sum()
        r2 = 1 - ((T - pred) ** 2).sum() / ss_tot if ss_tot > 0 else 1.0
        best_fill = by_b.loc[by_b["prove_per_sample"].idxmin()]
        with open("results/cost_model_timing.txt", "w") as f:
            f.write("ZKLR prove-time cost model (single-prover W=1 baseline)\n")
            f.write("=" * 60 + "\n")
            f.write("Constraint count is linear in batch, but PLONK proves over an\n")
            f.write("FFT domain padded to the next power of two — so prove time is a\n")
            f.write("STEP function of the padded domain. Per-sample cost is minimised\n")
            f.write("when the batch fills its 2^n domain.\n\n")
            f.write(f"{'batch':>6} {'constraints':>12} {'2^n domain':>12} "
                    f"{'fill%':>7} {'prove/batch(s)':>15} {'prove/sample(s)':>16}\n")
            for _, r in by_b.iterrows():
                f.write(f"{int(r.batch):>6} {int(r.constraints):>12} "
                        f"{int(r.pow2_domain):>12} {r.domain_fill:>6.1f}% "
                        f"{r.prove_per_batch:>15.3f} {r.prove_per_sample:>16.4f}\n")
            f.write(f"\nBest per-sample prove: batch={int(best_fill.batch)} "
                    f"({best_fill.domain_fill:.1f}% domain fill) "
                    f"-> {best_fill.prove_per_sample:.4f} s/sample\n")
            f.write(f"(naive linear fit T_prove≈{a:.2f}+{b:.3f}·B has R^2={r2:.3f} — "
                    f"poor, because the true cost is domain-quantised, not linear)\n")
        print(f"\nProve-time is domain-quantised (2^n FFT). Best per-sample: "
              f"batch={int(best_fill.batch)} @ {best_fill.domain_fill:.1f}% fill "
              f"-> {best_fill.prove_per_sample:.4f}s -> results/cost_model_timing.txt")
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

        # Prove-time amortization: per-sample cost vs batch, with the FFT-domain
        # boundaries that drive it. Twin axis overlays the domain-fill fraction.
        if a is not None:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.plot(by_b["batch"], by_b["prove_per_sample"], "o-", color="C0",
                    zorder=3, label="prove time / sample (W=1)")
            for _, r in by_b.iterrows():
                ax.annotate(f"2^{int(math.log2(r.pow2_domain))}\n{r.domain_fill:.0f}% full",
                            (r.batch, r.prove_per_sample), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=7, color="C3")
            ax.set_xlabel("batch size B")
            ax.set_ylabel("prove time per sample (s)", color="C0")
            ax.set_title("Prove-time amortization is driven by FFT-domain fill")
            ax.grid(alpha=0.3)
            ax2 = ax.twinx()
            ax2.bar(by_b["batch"], by_b["domain_fill"], width=4, alpha=0.15,
                    color="C3", label="domain fill %")
            ax2.set_ylabel("FFT-domain fill (%)", color="C3")
            ax2.set_ylim(0, 110)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig("results/fig_cost_model_timing.png", dpi=150)
            plt.close(fig)
            print("Wrote results/fig_cost_model_timing.png")
    except ImportError:
        print("matplotlib not available — skipped figures")


if __name__ == "__main__":
    main()
