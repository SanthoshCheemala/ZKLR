#!/usr/bin/env python3
"""
BTP Phase-2 Report Visualization Generator
Uses REAL HPC benchmark data (3K samples, 4 features, BN254, April 2026).

Output: Phase2_Report/figures/   (PNG @ 300 DPI + PDF)
        Phase2_Report/tables/    (LaTeX .tex files)
"""

import matplotlib.patches as mpatches
import os, math
import matplotlib
import matplotlib.pyplot as plt
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

BASE    = os.path.join(os.path.dirname(__file__), '..', 'Phase2_Report')
FIG_DIR = os.path.join(BASE, 'figures')
TAB_DIR = os.path.join(BASE, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

C = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

def save(name):
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close()
    print(f'  ✔ {name}')

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1 vs Phase 2 single-circuit (laptop, BN254, 1-feature)
P1 = dict(constraints=159_804, compile_ms=96,  setup_s=6.2,   prove_s=3.8, verify_ms=1.5, table=32_769)
P2 = dict(constraints=114_219, compile_ms=67,  setup_s=6.186, prove_s=4.127, verify_ms=1.599, table=20_481)

# Optimised circuit (tighter sigmoid ±10) from benchmark_test
OPT_IMPROVE = dict(constraints=32, compile=34, setup=47, prove=45, table=37)

# ── REAL HPC results (3 K samples, 4 features, BN254) ──────────────────────
# Worker scaling (1000 samples, batch=20, 4 features — from worker_benchmark.txt)
WORKERS      = [1, 2, 4, 6]
WALL_BY_WORKER = [0.531, 0.529, 0.538, 0.527]

# ── COMPREHENSIVE HPC SWEEP (3K Samples, BN254) ─────────────────────────────
SWEEP_WORKERS = [1, 4, 8, 16, 32, 64, 96]
SWEEP_BATCHES = [10, 20, 40, 80]
# Matrix of Wall-Clock/Sample (rows=batches, cols=workers)
SWEEP_WALL = np.array([
    [0.353, 0.289, 0.309, 0.319, 0.298, 0.310, 0.307],  # B=10
    [0.341, 0.285, 0.290, 0.298, 0.310, 0.292, 0.288],  # B=20
    [0.346, 0.307, 0.297, 0.786, 0.753, 0.786, 0.817],  # B=40
    [0.182, 0.160, 0.163, 0.468, 0.498, 0.511, 0.452]   # B=80
])

# Update batch configs to include the new ultimate winner
BATCH_CONFIGS = [
    {'label': 'batch=80\nworkers=4\n(Optimal HPC)', 'wall': 0.160, 'speedup': 12.5, 'setup_s': 11.68, 'constraints': 1_100_000},
    {'label': 'batch=40\nworkers=6\n(HPC,3K)', 'wall': 0.310, 'speedup': 6.5, 'setup_s': 10.27, 'constraints': 542_608},
    {'label': 'batch=40\nworkers=96\n(HPC,3K)','wall': 0.817, 'speedup': 2.4, 'setup_s': 10.68, 'constraints': 542_608},
    {'label': 'batch=40\nworkers=4\n(Laptop,100)', 'wall': 0.917, 'speedup': 2.2, 'setup_s': 27.34,'constraints': 542_608},
]

# =============================================================================
# Fig 00 — HPC Grid Sweep Heatmap (New!)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
cax = ax.imshow(SWEEP_WALL, cmap='YlOrRd', aspect='auto')
ax.set_xticks(np.arange(len(SWEEP_WORKERS)))
ax.set_yticks(np.arange(len(SWEEP_BATCHES)))
ax.set_xticklabels(SWEEP_WORKERS)
ax.set_yticklabels(SWEEP_BATCHES)
ax.set_xlabel('Number of Parallel Workers')
ax.set_ylabel('Batch Size')
ax.set_title('Wall-Clock latency per sample (s)\nSweet Spot: Batch=80, Workers=4')

# Annotate each cell with the numerical value
for i in range(len(SWEEP_BATCHES)):
    for j in range(len(SWEEP_WORKERS)):
        val = SWEEP_WALL[i, j]
        text_color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)

# Highlight the best square (B=80, W=4)
ax.add_patch(mpatches.Rectangle((1-0.5, 3-0.5), 1, 1, fill=False, edgecolor='blue', lw=3))

cbar = fig.colorbar(cax)
cbar.set_label('Seconds per sample')
save('fig_15_hpc_grid_sweep_heatmap')

# BN254 vs BLS12-377 (laptop, 100 samples, batch=40)
BN254_vs_BLS = dict(
    bn254  = dict(setup=27.34, wall=0.917, prove=2.233, proof_b=584),
    bls377 = dict(setup=89.87, wall=3.835, prove=4.144, proof_b=744),
)

print('\n[Generating all figures with real HPC data...]')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 01 — Constraints: Phase 1 vs Phase 2
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Phase 1\n(Baseline)', 'Phase 2\n(Optimised)']
vals   = [P1['constraints'], P2['constraints']]
bars   = ax.bar(labels, vals, color=[C[2], C[0]], width=0.45, zorder=3)
ax.set_ylabel('Circuit Constraints')
ax.set_title('Circuit Constraints: Phase 1 vs Phase 2')
ax.set_ylim(0, 195_000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+1500, f'{v:,}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.annotate('−29%', xy=(1, P2['constraints']), xytext=(1.35, 148_000),
            fontsize=10, color=C[1], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C[1], lw=1.5))
save('fig_01_constraints_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 02 — Overall % improvement across all metrics
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
metrics = ['Constraints\n−32%', 'Compile\n−34%', 'Setup\n−47%', 'Prove\n−45%', 'Table Size\n−37%']
vals    = [32, 34, 47, 45, 37]
bars    = ax.bar(metrics, vals, color=C[1], zorder=3)
ax.set_ylabel('Improvement (%)')
ax.set_title('Phase 2 Circuit Optimisation: Improvement vs Phase 1 Baseline')
ax.set_ylim(0, 58)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f'{v}%',
            ha='center', va='bottom', fontweight='bold', color=C[1])
save('fig_02_circuit_improvements')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 03 — Prove + Verify time: P1 vs P2 (single sample)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, unit, p1v, p2v, title in [
    (axes[0], 'Prove Time', 's', P1['prove_s'], P2['prove_s'], 'Prove Time per Sample'),
    (axes[1], 'Verify Time', 'ms', P1['verify_ms'], P2['verify_ms'], 'Verify Time per Sample'),
]:
    bars = ax.bar(['Phase 1', 'Phase 2'], [p1v, p2v], color=[C[2], C[0]], width=0.45, zorder=3)
    ax.set_ylabel(f'{metric} ({unit})')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    for bar, v in zip(bars, [p1v, p2v]):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.04, f'{v:.2f} {unit}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.tight_layout()
save('fig_03_prove_verify_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 04 — Wall-clock per sample: all HPC configs + laptop (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
labels = ['b=80\nw=4\n(Opt HPC)', 'b=20\nw=6\n(HPC)', 'b=40\nw=6\n(HPC)',
          'b=40\nw=96\n(HPC)', 'b=40\nw=4\n(Laptop)']
walls  = [0.160, 0.283, 0.310, 0.713, 0.917]
colors = [C[4], C[1], C[0], C[3], C[2]]
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(labels, walls, color=colors, zorder=3)
ax.set_ylabel('Wall-Clock Time per Sample (s)')
ax.set_title('Batch Configuration Comparison — Wall-Clock per Sample\n(BN254 PLONK, 4 Features)')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.set_ylim(0, 1.2)
for bar, v in zip(bars, walls):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.3f}s',
            ha='center', va='bottom', fontweight='bold')
ax.axhline(1/1, linestyle=':', color='grey', linewidth=1, label='1 s/sample reference')
ax.legend()
save('fig_04_wall_clock_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 05 — Speedup across configs
# ─────────────────────────────────────────────────────────────────────────────
speedups = [12.5, 7.1, 6.5, 2.8, 2.2]
fig, ax  = plt.subplots(figsize=(10, 4.5))
bars     = ax.bar(labels, speedups, color=colors, zorder=3)
ax.set_ylabel('Speedup vs. Single-Sample Sequential')
ax.set_title('Parallel Speedup Across Batch / Worker Configurations')
ax.set_ylim(0, 15)
ax.axhline(1, linestyle='--', color='grey', linewidth=1, label='1× (no speedup)')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
for bar, v in zip(bars, speedups):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v}×',
            ha='center', va='bottom', fontweight='bold')
ax.legend()
save('fig_05_speedup_comparison')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 06 — Worker scaling (1 K samples, batch=20)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(WORKERS, WALL_BY_WORKER, 's-', color=C[3], linewidth=2, markersize=8, zorder=3)
ax.fill_between(WORKERS, WALL_BY_WORKER, alpha=0.12, color=C[3])
ax.axvline(6, linestyle='--', color=C[1], linewidth=1.2, label='Optimal (6 workers, 0.527s)')
ax.set_xlabel('Number of Worker Goroutines')
ax.set_ylabel('Wall-Clock Time per Sample (s)')
ax.set_title('Worker Count Scaling (1 K Samples, Batch=20, 4 Features)')
ax.set_xticks(WORKERS)
ax.set_ylim(0.50, 0.56)
ax.legend()
ax.grid(linestyle='--', alpha=0.4)
for x, y in zip(WORKERS, WALL_BY_WORKER):
    ax.annotate(f'{y}s', (x, y), textcoords='offset points', xytext=(6, 5), fontsize=9)
save('fig_06_worker_scaling')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 07 — Setup time: batch=20 vs batch=40 (HPC + laptop)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
setups = [11.68, 5.28, 10.27, 10.18, 27.34]
bars   = ax.bar(labels, setups, color=colors, zorder=3)
ax.set_ylabel('Setup Time (s)')
ax.set_title('One-Time Setup / SRS Generation Time by Configuration')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
for bar, v in zip(bars, setups):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f'{v:.1f}s',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
save('fig_07_setup_time')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 08 — BN254 vs BLS12-377 grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
metrics  = ['Setup (s)', 'Wall-clock/\nSample (s)', 'Prove/\nSample (s)', 'Proof Size\n(bytes÷10)']
bn254_v  = [27.34, 0.917, 2.233, 584/10]
bls377_v = [89.87, 3.835, 4.144, 744/10]
x = np.arange(len(metrics)); w = 0.32
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x-w/2, bn254_v,  w, label='BN254 (selected)',  color=C[0], zorder=3)
ax.bar(x+w/2, bls377_v, w, label='BLS12-377 (tested)', color=C[2], zorder=3)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title('BN254 vs. BLS12-377 — Key Metrics (100 Samples, Batch=40, Laptop)')
ax.set_ylabel('Value (see axis labels)')
ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
save('fig_08_bn254_vs_bls377')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 09 — Throughput (samples/sec) across configs
# ─────────────────────────────────────────────────────────────────────────────
throughputs = [1/w for w in walls]
fig, ax     = plt.subplots(figsize=(10, 4.5))
bars        = ax.bar(labels, throughputs, color=colors, zorder=3)
ax.set_ylabel('Throughput (samples / second)')
ax.set_title('Proving Throughput by Configuration')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
for bar, v in zip(bars, throughputs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.04, f'{v:.2f}',
            ha='center', va='bottom', fontweight='bold')
save('fig_09_throughput')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Scalability: Samples vs total wall-clock time (HPC, batch=20, w=6)
# ─────────────────────────────────────────────────────────────────────────────
# 0.160s/sample on HPC; real point at 3000 samples
samples   = [100, 500, 1000, 2000, 3000, 5000, 10000]
total_hpc = [s * 0.160 for s in samples]
total_lap = [s * 0.917 for s in samples]
fig, ax   = plt.subplots(figsize=(8, 4.5))
ax.plot(samples, total_hpc, 'o-', color=C[4], linewidth=2, markersize=6, label='HPC (batch=80, w=4) — 0.160s/samp')
ax.plot(samples, total_lap, 's--', color=C[2], linewidth=2, markersize=6, label='Laptop (batch=40, w=4) — 0.917s/samp')
ax.scatter([3000], [3000*0.160], s=120, zorder=5, color=C[4], edgecolors='black', linewidth=1)
ax.annotate('Measured\n(HPC, 3K)', (3000, 3000*0.160), xytext=(3200, 650),
            fontsize=9, color=C[4], arrowprops=dict(arrowstyle='->', color=C[4]))
ax.scatter([100], [100*0.917], s=120, zorder=5, color=C[2], edgecolors='black', linewidth=1)
ax.annotate('Measured\n(Laptop)', (100, 100*0.917), xytext=(300, 250),
            fontsize=9, color=C[2], arrowprops=dict(arrowstyle='->', color=C[2]))
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Total Wall-Clock Time (s)')
ax.set_title('Scalability: Dataset Size vs. Total Prediction Time')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}s'))
save('fig_10_scalability_curve')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — Proof size: constant regardless of batch/config
# ─────────────────────────────────────────────────────────────────────────────
cfg_labels = ['b=80 w=4\n(HPC)', 'b=20 w=6\n(HPC)', 'b=40 w=6\n(HPC)', 'b=40 w=96\n(HPC)', 'b=40 w=4\n(Laptop)']
fig, ax    = plt.subplots(figsize=(8, 4))
ax.barh(cfg_labels, [584]*5, color=C[0], zorder=3, label='BN254 (584 B)')
ax.barh(['BLS12-377 (laptop)'], [744], color=C[2], zorder=3, label='BLS12-377 (744 B)')
ax.set_xlabel('Proof Size (bytes)')
ax.set_title('Proof Size is Constant Across All Configurations')
ax.set_xlim(0, 900)
ax.axvline(584, linestyle='--', color=C[0], linewidth=1, alpha=0.5)
ax.legend(); ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
for i, v in enumerate([584]*4 + [744]):
    ax.text(v+10, i, f'{v} B', va='center', fontweight='bold', fontsize=9)
save('fig_11_proof_size')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 — Sigmoid precision error distribution
# ─────────────────────────────────────────────────────────────────────────────
z_vals = np.linspace(-10, 10, 10000)
def sig_float(z): return 1/(1+math.exp(-z))
def sig_circuit(z):
    idx = max(0, min(int((z+10)*(1<<10)), 20480))
    return round(sig_float(idx/(1<<10)-10)*(1<<16))/(1<<16)
errors = [abs(sig_circuit(float(z)) - sig_float(float(z))) for z in z_vals]
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(z_vals, errors, color=C[4], linewidth=0.7, alpha=0.85)
ax.axhline(np.mean(errors), color=C[1], linewidth=1.4, linestyle='--',
           label=f'Mean: {np.mean(errors):.6f}')
ax.axhline(max(errors), color=C[2], linewidth=1.4, linestyle='-.',
           label=f'Max: {max(errors):.6f}')
ax.set_xlabel('Input Z'); ax.set_ylabel('Absolute Error')
ax.set_title('Sigmoid Lookup Table: Approximation Error vs. Exact Float')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)
save('fig_12_sigmoid_precision')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 13 — Radar chart: Phase 2 improvements
# ─────────────────────────────────────────────────────────────────────────────
cats   = ['Constraints\n−32%', 'Compile\n−34%', 'Setup\n−47%', 'Prove\n−45%', 'Table\n−37%']
vals13 = [32, 34, 47, 45, 37]
N      = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
vp     = vals13 + vals13[:1]
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.fill(angles, vp, alpha=0.25, color=C[0])
ax.plot(angles, vp, 'o-', linewidth=2, color=C[0])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=10)
ax.set_ylim(0, 60); ax.set_yticks([10,20,30,40,50])
ax.set_yticklabels(['10%','20%','30%','40%','50%'], size=8)
ax.set_title('Phase 2 Circuit Optimisation Summary\n(% Improvement vs Phase 1)',
             pad=20, fontsize=12, fontweight='bold')
save('fig_13_radar_improvements')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 14 — HPC vs Laptop head-to-head (best config each)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
comparisons = [
    ('Wall-clock / Sample (s)', 0.283, 0.917, 'Lower is better'),
    ('Speedup vs Sequential',    7.1,   2.2,  'Higher is better'),
    ('Setup Time (s)',           5.28,  27.34, 'Lower is better'),
]
for ax, (ylabel, hpc_v, lap_v, note) in zip(axes, comparisons):
    bars = ax.bar(['HPC\n(best config)', 'Laptop\n(best config)'],
                  [hpc_v, lap_v], color=[C[1], C[2]], width=0.5, zorder=3)
    ax.set_ylabel(ylabel); ax.set_title(f'{ylabel}\n({note})', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    for bar, v in zip(bars, [hpc_v, lap_v]):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.04, f'{v}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.suptitle('HPC vs. Laptop: Best Configuration Head-to-Head (BN254, 3K/100 Samples)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save('fig_14_hpc_vs_laptop')

print('\n[All figures saved to Phase2_Report/figures/]')

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Tables
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Generating LaTeX tables...]')

t1 = r"""\begin{table}[H]
\centering
\caption{Performance Comparison: Phase 1 (Baseline) vs.\ Phase 2 (Optimised)}
\label{tab:phase_comparison}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{Phase 1} & \textbf{Phase 2} & \textbf{Improvement} \\
\hline
Circuit Constraints        & 159,804 & 114,219 & $-$29\% \\
Sigmoid Table Entries      & 32,769  & 20,481  & $-$37\% \\
Compile Time               & 96 ms   & 67 ms   & $-$30\% \\
1-Feature Setup Time       & 6.2 s   & 6.2 s   & $\approx$0 \\
Single Prove Time          & 3.8 s   & 4.1 s   & --- \\
Verify Time                & 1.5 ms  & 1.6 ms  & $\approx$0 \\
Proof Size                 & 584 B   & 584 B   & Constant \\
\hline
\end{tabular}
\end{table}
"""

t2 = r"""\begin{table}[H]
\centering
\caption{HPC Benchmark Results — 3,000 Samples, 4 Features, BN254 PLONK}
\label{tab:hpc_benchmark}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
\textbf{Batch} & \textbf{Workers} & \textbf{Batches} & \textbf{Setup (s)} & \textbf{Wall/Sample (s)} & \textbf{Prove/Samp (s)} & \textbf{Speedup} \\
\hline
20 & 6  & 150 & 5.28  & \textbf{0.283} & 1.663  & \textbf{7.1$\times$} \\
40 & 6  & 75  & 10.27 & 0.310 & 1.789  & 6.5$\times$ \\
40 & 96 & 75  & 10.18 & 0.713 & 10.694 & 2.8$\times$ \\
\hline
\multicolumn{7}{|l|}{\textit{Accuracy: 70.67\% (2120/3000 correct). Proof size: 584 bytes per batch (constant).}} \\
\hline
\end{tabular}
\end{table}
"""

t3 = r"""\begin{table}[H]
\centering
\caption{Experimental Setup}
\label{tab:exp_setup}
\begin{tabular}{|l|l|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
Local Machine  & Apple MacBook Air (M-Series, 8-core, 16 GB) \\
HPC Machine    & Linux cluster (96-core, accessed via SSH) \\
Language       & Go 1.22+ \\
ZK Library     & gnark v0.11.0 \\
ZK Backend     & PLONK \\
Elliptic Curve & BN254 \\
Dataset (local) & \texttt{public\_100.csv} (100 samples, 2 features) \\
Dataset (HPC)   & \texttt{test\_3000.csv} (3,000 samples, 4 features) \\
HPC Runs       & batch=\{20,40\}, workers=\{6,96\} \\
\hline
\end{tabular}
\end{table}
"""

t4 = r"""\begin{table}[H]
\centering
\caption{Circuit Architecture Summary}
\label{tab:circuit_arch}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Component} & \textbf{Choice} & \textbf{Rationale} \\
\hline
Proof System   & PLONK (gnark)    & Supports lookup tables; universal SRS \\
Curve          & BN254            & Fast proving; 128-bit security; EVM-native \\
Scaling        & $2^{32}$ fixed-point & Float $\to$ integer for finite field \\
Sigmoid        & Lookup table (20,481 entries) & $O(1)$; avoids Taylor series \\
Sigmoid Range  & $\pm 10$ (shifted $+10$) & 37\% smaller vs $\pm 16$ baseline \\
Truncation     & Remainder hint ($2^{22}$) & Avoids modular division \\
Clamping       & \texttt{api.Cmp} + \texttt{api.Select} & Edge-case safety \\
Feature Support & Dynamic slices ($N$-feature) & Supports any model size \\
Batch Mode     & Worker-pool goroutines & Amortises setup; full CPU use \\
\hline
\end{tabular}
\end{table}
"""

for fname, content in [
    ('table_01_performance.tex', t1),
    ('table_02_hpc_benchmark.tex', t2),
    ('table_03_exp_setup.tex',   t3),
    ('table_04_circuit.tex',     t4),
]:
    with open(os.path.join(TAB_DIR, fname), 'w') as f: f.write(content)
    print(f'  ✔ {fname}')

print('\n✅ Done! All figures and tables generated.\n')
