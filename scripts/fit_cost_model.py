#!/usr/bin/env python3
"""Fit the amortization cost model T(B) = a + b·B (F1 / Table inputs).

Inputs:
  --constraints CSV from cmd/constraint_scan (batch,features,mode,constraints,compile_ms)
  --timings     optional CSV with columns batch,prove_seconds (one row per run;
                repeated batch values are aggregated by median)

Outputs:
  results/cost_model.txt — fitted coefficients + R² + per-point residuals
  results/fig_cost_model.png — measured points vs fitted line (if matplotlib)
"""

import argparse
import os

import numpy as np
import pandas as pd


def fit_linear(x, y):
    A = np.vstack([np.ones_like(x, dtype=float), x.astype(float)]).T
    coef, *_ = np.linalg.lstsq(A, y.astype(float), rcond=None)
    a, b = coef
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", default="results/constraint_scaling_f9.csv")
    ap.add_argument("--timings", default=None)
    ap.add_argument("--out", default="results/cost_model.txt")
    ap.add_argument("--fig", default="results/fig_cost_model.png")
    args = ap.parse_args()

    lines = []

    dfc = pd.read_csv(args.constraints)
    B = dfc["batch"].values
    C = dfc["constraints"].values
    a, b, r2, pred = fit_linear(B, C)

    lines.append("ZKLR Amortization Cost Model")
    lines.append("=" * 60)
    lines.append(f"constraints source: {args.constraints} "
                 f"(features={dfc['features'].iloc[0]}, mode={dfc['mode'].iloc[0]})")
    lines.append("")
    lines.append(f"Constraints(B) = {a:,.0f} + {b:,.1f} * B    (R^2 = {r2:.6f})")
    lines.append(f"  shared cost (table + commitment): {a:,.0f} constraints")
    lines.append(f"  marginal cost per sample:         {b:,.1f} constraints")
    lines.append(f"  amortization: at B=80 each sample carries "
                 f"{(a / 80 + b):,.0f} constraints vs {a + b:,.0f} unbatched "
                 f"({(a + b) / (a / 80 + b):.1f}x reduction)")
    lines.append("")
    lines.append(f"{'batch':>6} {'measured':>12} {'fitted':>12} {'residual':>10}")
    for bb, cc, pp in zip(B, C, pred):
        lines.append(f"{bb:>6} {cc:>12,} {pp:>12,.0f} {cc - pp:>10,.0f}")

    tfit = None
    if args.timings and os.path.exists(args.timings):
        dft = pd.read_csv(args.timings)
        agg = dft.groupby("batch")["prove_seconds"].median().reset_index()
        ta, tb, tr2, tpred = fit_linear(agg["batch"].values, agg["prove_seconds"].values)
        tfit = (agg, ta, tb, tr2, tpred)
        lines.append("")
        lines.append(f"ProveTime(B) = {ta:.3f}s + {tb:.4f}s * B    (R^2 = {tr2:.6f})")
        lines.append(f"  per-sample marginal prove time: {tb * 1000:.1f} ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2 if tfit is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.2))
        ax0 = axes[0] if ncols == 2 else axes
        ax0.scatter(B, C / 1e3, label="measured", zorder=3)
        xs = np.linspace(0, B.max() * 1.05, 100)
        ax0.plot(xs, (a + b * xs) / 1e3, "--",
                 label=f"fit: {a/1e3:.1f}K + {b/1e3:.2f}K·B  (R²={r2:.4f})")
        ax0.set_xlabel("batch size B")
        ax0.set_ylabel("constraints (×1000)")
        ax0.set_title("Shared-lookup amortization: constraints vs batch size")
        ax0.legend()
        ax0.grid(alpha=0.3)

        if tfit is not None:
            agg, ta, tb, tr2, _ = tfit
            ax1 = axes[1]
            ax1.scatter(agg["batch"], agg["prove_seconds"], zorder=3, label="measured (median)")
            ax1.plot(xs, ta + tb * xs, "--",
                     label=f"fit: {ta:.2f}s + {tb*1000:.1f}ms·B  (R²={tr2:.4f})")
            ax1.set_xlabel("batch size B")
            ax1.set_ylabel("prove time (s)")
            ax1.set_title("Prove time vs batch size")
            ax1.legend()
            ax1.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
        print(f"Wrote {args.fig}")
    except ImportError:
        print("matplotlib not available — skipped figure")


if __name__ == "__main__":
    main()
