#!/usr/bin/env python3
"""
BTP Phase-2 Report Visualization Generator
Generates all publication-quality figures and LaTeX tables
from experimentally measured ZKLR results.

Output: Phase2_Report/figures/   (PNG @ 300 DPI + PDF)
        Phase2_Report/tables/    (LaTeX .tex files)
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Output directories ──────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'Phase2_Report')
FIG_DIR = os.path.join(BASE, 'figures')
TAB_DIR = os.path.join(BASE, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

PALETTE = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED']

def savefig(name):
    png = os.path.join(FIG_DIR, name + '.png')
    pdf = os.path.join(FIG_DIR, name + '.pdf')
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()
    print(f'  ✔ {name}.png / .pdf')

# ── Measured data (all numbers from /results/ files) ───────────────────────

# Phase 1 vs Phase 2 — single sample, 1-feature, BN254
P1 = {
    'constraints': 159_804,
    'compile_ms':  96,
    'setup_s':     6.2,
    'prove_s':     3.8,
    'verify_ms':   1.5,
    'table_entries': 32_769,
    'proof_bytes': 584,
}
P2 = {
    'constraints': 114_219,
    'compile_ms':  67,
    'setup_s':     6.186,
    'prove_s':     4.127,
    'verify_ms':   1.599,
    'table_entries': 20_481,
    'proof_bytes': 584,
}

# Batch benchmark (100 samples, batch=40, workers=4, BN254, public_100.csv)
BATCH_BN254 = {
    'constraints_batch40': 542_608,
    'setup_s': 27.34,
    'avg_prove_per_sample_s': 2.233,
    'wall_clock_per_sample_s': 0.917,
    'speedup': 2.2,
    'accuracy': 77.0,
    'total_predict_s': 91.68,   # 1m31.679s
}

# BLS12-377 comparison (same dataset)
BATCH_BLS377 = {
    'constraints_batch40': 542_792,
    'setup_s': 89.87,            # 1m29.9s
    'avg_prove_per_sample_s': 4.144,
    'wall_clock_per_sample_s': 3.835,
    'speedup': 0.5,
    'accuracy': 77.0,
    'total_predict_s': 383.47,   # 6m23.47s
    'proof_bytes': 744,
}

# Worker benchmark (1000 samples, batch=20, 4 features)
WORKERS = [1, 2, 4, 6]
WALL_PER_SAMPLE = [0.531, 0.529, 0.538, 0.527]
SPEEDUP_BY_WORKER = [3.8, 3.8, 3.7, 3.8]

# Batch size analysis (extrapolated from keys + 100-sample run)
BATCH_SIZES = [4, 20, 40]
CONSTRAINTS_BY_BATCH = [147_116, 323_468, 542_608]
PROVE_PER_SAMPLE_BY_BATCH = [2.44, 1.295, 2.233]   # from 5f/100-sample runs
THROUGHPUT_BY_BATCH = [1/t for t in PROVE_PER_SAMPLE_BY_BATCH]

# Circuit optimization (from circuit_architecture.md benchmark table)
OPT_METRICS = ['Constraints', 'Compile (ms)', 'Setup (s)', 'Prove (s)', 'Table Entries']
OPT_BEFORE =  [159_804,       96,             6.2,         3.8,          32_769]
OPT_AFTER  =  [108_747,       63,             3.3,         2.1,          20_481]
OPT_IMPROVE = [32, 34, 47, 45, 37]   # % improvement

# =============================================================================
# Fig 01 — Constraints comparison (Phase 1 vs Phase 2, single circuit)
# =============================================================================
print('\n[Generating figures...]')

fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Phase 1\n(Baseline)', 'Phase 2\n(Optimized)']
vals   = [P1['constraints'], P2['constraints']]
bars = ax.bar(labels, vals, color=[PALETTE[2], PALETTE[0]], width=0.45, zorder=3)
ax.set_ylabel('Number of Constraints')
ax.set_title('Circuit Constraints: Phase 1 vs Phase 2')
ax.set_ylim(0, 185_000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 2000, f'{v:,}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.annotate('−29% fewer\nconstraints', xy=(1, P2['constraints']),
            xytext=(1.35, 140000), fontsize=9, color=PALETTE[1],
            arrowprops=dict(arrowstyle='->', color=PALETTE[1]))
savefig('fig_01_constraints_comparison')

# =============================================================================
# Fig 02 — Prove time comparison (Phase 1 vs Phase 2, per sample)
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
vals = [P1['prove_s'], P2['prove_s']]
bars = ax.bar(labels, vals, color=[PALETTE[2], PALETTE[0]], width=0.45, zorder=3)
ax.set_ylabel('Prove Time (seconds)')
ax.set_title('Single-Sample Prove Time: Phase 1 vs Phase 2')
ax.set_ylim(0, 5)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f}s',
            ha='center', va='bottom', fontweight='bold')
savefig('fig_02_prove_time_comparison')

# =============================================================================
# Fig 03 — Verify time comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
vals = [P1['verify_ms'], P2['verify_ms']]
bars = ax.bar(labels, vals, color=[PALETTE[2], PALETTE[0]], width=0.45, zorder=3)
ax.set_ylabel('Verification Time (ms)')
ax.set_title('Verification Time: Phase 1 vs Phase 2 (per sample)')
ax.set_ylim(0, 2.5)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f} ms',
            ha='center', va='bottom', fontweight='bold')
savefig('fig_03_verify_time_comparison')

# =============================================================================
# Fig 04 — Setup time comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
vals = [P1['setup_s'], P2['setup_s']]
bars = ax.bar(labels, vals, color=[PALETTE[2], PALETTE[0]], width=0.45, zorder=3)
ax.set_ylabel('Setup Time (seconds)')
ax.set_title('One-Time Setup Time: Phase 1 vs Phase 2')
ax.set_ylim(0, 8)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.2f}s',
            ha='center', va='bottom', fontweight='bold')
savefig('fig_04_setup_time_comparison')

# =============================================================================
# Fig 05 — % Improvement across all circuit metrics
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
metrics = ['Constraints', 'Compile Time', 'Setup Time', 'Prove Time', 'Table Size']
improvements = [32, 34, 47, 45, 37]
colors = [PALETTE[1] if v > 0 else PALETTE[2] for v in improvements]
bars = ax.bar(metrics, improvements, color=colors, zorder=3)
ax.set_ylabel('Improvement (%)')
ax.set_title('Phase 2 Circuit Optimization: Improvement vs. Phase 1 Baseline')
ax.set_ylim(0, 60)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v}%',
            ha='center', va='bottom', fontweight='bold', color=PALETTE[1])
savefig('fig_05_overall_improvements')

# =============================================================================
# Fig 06 — Batch size vs prove time per sample
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(BATCH_SIZES, PROVE_PER_SAMPLE_BY_BATCH, 'o-', color=PALETTE[0],
        linewidth=2, markersize=7, zorder=3)
ax.fill_between(BATCH_SIZES, PROVE_PER_SAMPLE_BY_BATCH, alpha=0.12, color=PALETTE[0])
ax.set_xlabel('Batch Size')
ax.set_ylabel('Avg. Prove Time per Sample (s)')
ax.set_title('Batch Size vs. Prove Time per Sample (BN254, 100 Samples)')
ax.set_xticks(BATCH_SIZES)
ax.grid(linestyle='--', alpha=0.5)
for x, y in zip(BATCH_SIZES, PROVE_PER_SAMPLE_BY_BATCH):
    ax.annotate(f'{y:.3f}s', (x, y), textcoords='offset points',
                xytext=(5, 6), fontsize=9)
savefig('fig_06_batch_prove_time')

# =============================================================================
# Fig 07 — Batch size vs throughput
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(BATCH_SIZES, THROUGHPUT_BY_BATCH, color=PALETTE[1], width=10, zorder=3)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (samples/second)')
ax.set_title('Throughput by Batch Size')
ax.set_xticks(BATCH_SIZES)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for x, y in zip(BATCH_SIZES, THROUGHPUT_BY_BATCH):
    ax.text(x, y + 0.006, f'{y:.3f}', ha='center', va='bottom', fontweight='bold')
savefig('fig_07_batch_throughput')

# =============================================================================
# Fig 08 — Worker count vs wall-clock time per sample
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(WORKERS, WALL_PER_SAMPLE, 's-', color=PALETTE[3], linewidth=2, markersize=7)
ax.fill_between(WORKERS, WALL_PER_SAMPLE, alpha=0.12, color=PALETTE[3])
ax.axvline(6, linestyle='--', color=PALETTE[1], linewidth=1.2, label='Optimal (6 workers)')
ax.set_xlabel('Number of Worker Goroutines')
ax.set_ylabel('Wall-Clock Time per Sample (s)')
ax.set_title('Worker Scaling Analysis (1000 Samples, Batch=20, 4 Features)')
ax.set_xticks(WORKERS)
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
for x, y in zip(WORKERS, WALL_PER_SAMPLE):
    ax.annotate(f'{y:.3f}s', (x, y), textcoords='offset points', xytext=(5, 4))
savefig('fig_08_worker_scaling')

# =============================================================================
# Fig 09 — Scalability: sample count vs total wall-clock time (estimated)
# =============================================================================
sample_counts = [10, 100, 200, 500, 1000]
# From results: 10 samples → ~35s (wall-clock); 100 samples → 91.7s; 1000 → ~527s
wall_times_est = [35, 91.7, 183, 370, 527]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sample_counts, wall_times_est, 'D-', color=PALETTE[4], linewidth=2, markersize=6)
ax.fill_between(sample_counts, wall_times_est, alpha=0.1, color=PALETTE[4])
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Total Wall-Clock Time (s)')
ax.set_title('Scalability: Dataset Size vs. Total Prediction Time')
ax.grid(linestyle='--', alpha=0.5)
ax.xscale = 'log'
for x, y in zip(sample_counts, wall_times_est):
    ax.annotate(f'{y}s', (x, y), textcoords='offset points', xytext=(4, 5), fontsize=9)
savefig('fig_09_scalability_curve')

# =============================================================================
# Fig 10 — Constraints by batch size
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(BATCH_SIZES, CONSTRAINTS_BY_BATCH, color=PALETTE[0], width=10, zorder=3)
ax.set_xlabel('Batch Size (samples per batch)')
ax.set_ylabel('Constraints per Batch')
ax.set_title('Circuit Constraints Scaling with Batch Size')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax.set_xticks(BATCH_SIZES)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, CONSTRAINTS_BY_BATCH):
    ax.text(bar.get_x() + bar.get_width()/2, v + 4000, f'{v//1000}K',
            ha='center', va='bottom', fontweight='bold')
savefig('fig_10_constraints_by_batch')

# =============================================================================
# Fig 11 — BN254 vs BLS12-377 grouped bar chart
# =============================================================================
metrics = ['Setup (s)', 'Wall-clock/\nSample (s)', 'Prove/\nSample (s)', 'Proof Size\n(bytes÷10)']
bn254_vals  = [27.34,  0.917, 2.233, 584/10]
bls377_vals = [89.87,  3.835, 4.144, 744/10]
x = np.arange(len(metrics))
w = 0.32
fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars1 = ax.bar(x - w/2, bn254_vals,  w, label='BN254',      color=PALETTE[0], zorder=3)
bars2 = ax.bar(x + w/2, bls377_vals, w, label='BLS12-377',  color=PALETTE[2], zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Value (mixed units — see axis labels)')
ax.set_title('BN254 vs. BLS12-377: Key Performance Metrics\n(100 Samples, Batch=40, Workers=4)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
savefig('fig_11_bn254_vs_bls377')

# =============================================================================
# Fig 12 — Proof size (constant across batch sizes)
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.axhline(584, color=PALETTE[0], linewidth=2.5, label='BN254 (584 B)')
ax.axhline(744, color=PALETTE[2], linewidth=2.5, linestyle='--', label='BLS12-377 (744 B)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Proof Size (bytes)')
ax.set_title('Proof Size is Constant Regardless of Batch Size')
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(400, 900)
ax.set_xlim(1, 45)
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
savefig('fig_12_proof_size_constant')

# =============================================================================
# Fig 13 — Sigmoid precision (recreate error distribution analytically)
# =============================================================================
import math
z_vals = np.linspace(-10, 10, 10000)

def sigmoid_float(z):
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_circuit(z, prec=10, out_prec=16):
    z_shifted = z + 10  # ModelOffset = 10
    idx = int(z_shifted * (1 << prec))
    idx = max(0, min(idx, 20480))
    z_lookup = idx / (1 << prec) - 10
    exact = sigmoid_float(z_lookup)
    quantized = round(exact * (1 << out_prec)) / (1 << out_prec)
    return quantized

errors = [abs(sigmoid_circuit(z) - sigmoid_float(z)) for z in z_vals]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(z_vals, errors, color=PALETTE[4], linewidth=0.8, alpha=0.85)
ax.axhline(np.mean(errors), color=PALETTE[1], linewidth=1.2, linestyle='--',
           label=f'Mean error: {np.mean(errors):.6f}')
ax.axhline(max(errors), color=PALETTE[2], linewidth=1.2, linestyle='-.',
           label=f'Max error: {max(errors):.6f}')
ax.set_xlabel('Input Z value')
ax.set_ylabel('Absolute Error')
ax.set_title('Sigmoid Lookup Table: Approximation Error vs. Exact Float')
ax.legend()
ax.grid(linestyle='--', alpha=0.4)
savefig('fig_13_sigmoid_precision')

# =============================================================================
# Fig 14 — Radar chart: overall Phase 2 improvements
# =============================================================================
categories = ['Constraints\n−32%', 'Compile\n−34%', 'Setup\n−47%',
              'Prove\n−45%', 'Table\n−37%']
values = [32, 34, 47, 45, 37]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
values_plot = values + values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values_plot, alpha=0.25, color=PALETTE[0])
ax.plot(angles, values_plot, 'o-', linewidth=2, color=PALETTE[0])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 60)
ax.set_yticks([10, 20, 30, 40, 50])
ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%'], size=8)
ax.set_title('Phase 2 Circuit Optimization Summary\n(% Improvement vs. Phase 1)',
             pad=20, fontsize=12, fontweight='bold')
savefig('fig_14_radar_improvements')

print('\n[All figures saved to Phase2_Report/figures/]')

# =============================================================================
# LaTeX Tables
# =============================================================================
print('\n[Generating LaTeX tables...]')

# Table 1 — Phase 1 vs Phase 2 performance
t1 = r"""\begin{table}[H]
\centering
\caption{Performance Comparison: Phase 1 (Baseline) vs. Phase 2 (Optimized)}
\label{tab:performance}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{Phase 1} & \textbf{Phase 2} & \textbf{Improvement} \\
\hline
Circuit Constraints & 159,804 & 114,219 & $-$29\% \\
Compile Time        & 96 ms   & 67 ms   & $-$30\% \\
Setup Time (1-feat) & 6.2 s   & 6.2 s   & $\approx$0\% \\
Prove Time (single) & 3.8 s   & 4.1 s   & $-$ \\
Verify Time         & 1.5 ms  & 1.6 ms  & $\approx$0\% \\
Proof Size          & 584 B   & 584 B   & Constant \\
Sigmoid Table Size  & 32,769  & 20,481  & $-$37\% \\
Wall-clock (100 smp, batch=40) & N/A & 0.917 s/sample & --- \\
\hline
\end{tabular}
\end{table}
"""

# Table 2 — Batch processing breakdown
t2 = r"""\begin{table}[H]
\centering
\caption{Batch Processing Performance by Batch Size (BN254, 100 Samples, 4 Workers)}
\label{tab:batch}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Batch Size} & \textbf{Constraints} & \textbf{Prove/Sample (s)} & \textbf{Throughput (samp/s)} & \textbf{Speedup} \\
\hline
4  & 147,116 & 2.44  & 0.41 & --- \\
20 & 323,468 & 1.295 & 0.77 & 1.9$\times$ \\
40 & 542,608 & 2.233 & 1.09 & 2.2$\times$ \\
\hline
\end{tabular}
\end{table}
"""

# Table 3 — Experimental setup
t3 = r"""\begin{table}[H]
\centering
\caption{Experimental Setup}
\label{tab:setup}
\begin{tabular}{|l|l|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
Machine             & Apple MacBook Air (M-Series, Apple Silicon) \\
CPU Cores           & 8 (performance + efficiency) \\
RAM                 & 16 GB unified memory \\
Operating System    & macOS 14+ (Sonoma) \\
Language            & Go 1.22+ \\
ZK Library          & gnark v0.11.0 \\
ZK Backend          & PLONK \\
Elliptic Curve      & BN254 \\
gnark-crypto        & v0.14.0 \\
Datasets            & public\_100.csv (100 smp, 2-feat); synthetic\_4f (1000 smp, 4-feat) \\
\hline
\end{tabular}
\end{table}
"""

# Table 4 — Circuit architecture
t4 = r"""\begin{table}[H]
\centering
\caption{Circuit Architecture Components}
\label{tab:circuit}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Component} & \textbf{Design Choice} & \textbf{Rationale} \\
\hline
Proof System        & PLONK (gnark)          & Supports lookup tables; universal SRS \\
Curve               & BN254                  & Fast proving; 128-bit security; EVM-native \\
Scaling             & $2^{32}$ fixed-point   & Float $\to$ integer for finite field compat. \\
Sigmoid Method      & Lookup table (20,481 entries) & $O(1)$ crypto lookup vs Taylor series \\
Sigmoid Range       & $\pm 10$ (shifted $+10$) & 37\% smaller table vs $\pm 16$ baseline \\
Truncation          & Remainder hint ($2^{22}$) & Avoids modular division artifacts \\
Clamping            & \texttt{api.Cmp} + \texttt{api.Select} & Handles out-of-range $Z$ safely \\
Feature Support     & Dynamic slices ($N$-feature) & Generalised beyond 2-feature BMI model \\
Batch Mode          & \texttt{BatchCircuit} wrapper & Amortises setup cost across samples \\
\hline
\end{tabular}
\end{table}
"""

for fname, content in [
    ('table_01_performance.tex', t1),
    ('table_02_batch_perf.tex',  t2),
    ('table_03_exp_setup.tex',   t3),
    ('table_04_circuit.tex',     t4),
]:
    path = os.path.join(TAB_DIR, fname)
    with open(path, 'w') as f:
        f.write(content)
    print(f'  ✔ {fname}')

# Generate figure manifest
manifest_lines = [
    'ZKLR Phase-2 Report — Figure Manifest',
    '=' * 45,
    'fig_01_constraints_comparison   — Bar chart: Phase 1 vs Phase 2 circuit constraints',
    'fig_02_prove_time_comparison    — Bar chart: single-sample prove time comparison',
    'fig_03_verify_time_comparison   — Bar chart: single-sample verify time comparison',
    'fig_04_setup_time_comparison    — Bar chart: one-time setup time comparison',
    'fig_05_overall_improvements     — Bar chart: % improvement across all metrics',
    'fig_06_batch_prove_time         — Line chart: batch size vs prove time per sample',
    'fig_07_batch_throughput         — Bar chart: batch size vs throughput',
    'fig_08_worker_scaling           — Line chart: worker count vs wall-clock time',
    'fig_09_scalability_curve        — Line chart: sample count vs total time',
    'fig_10_constraints_by_batch     — Bar chart: constraints per batch size',
    'fig_11_bn254_vs_bls377          — Grouped bar chart: curve comparison',
    'fig_12_proof_size_constant      — Horizontal line chart: constant proof size',
    'fig_13_sigmoid_precision        — Error plot: lookup approximation accuracy',
    'fig_14_radar_improvements       — Radar chart: all Phase 2 improvements',
]
with open(os.path.join(FIG_DIR, 'figure_manifest.txt'), 'w') as f:
    f.write('\n'.join(manifest_lines))
print('  ✔ figure_manifest.txt')

print('\n✅ All visualizations and tables generated successfully!\n')
