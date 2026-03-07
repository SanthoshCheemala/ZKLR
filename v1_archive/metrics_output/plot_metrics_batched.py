#!/usr/bin/env python3
"""
Generate PNG visualizations for batched ZK-LR metrics
Panel presentation ready charts
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

METRICS_DIR = Path(__file__).parent
plt.style.use('seaborn-v0_8-darkgrid')

def plot_comparison():
    """Individual vs Batched speedup comparison"""
    df = pd.read_csv(METRICS_DIR / 'comparison_batched.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time comparison
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df['Individual Time (s)'], width, label='Individual', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, df['Batched Time (s)'], width, label='Batched', color='#27ae60', alpha=0.8)
    ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Proving Time: Individual vs Batched', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Dataset Size'])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Speedup chart
    speedup_vals = [float(s.replace('x', '')) for s in df['Speedup']]
    bars = ax2.bar(df['Dataset Size'], speedup_vals, color='#3498db', alpha=0.8, edgecolor='navy')
    ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title('Batching Speedup (20 samples/batch)', fontsize=14, fontweight='bold')
    ax2.axhline(y=1, color='red', linestyle='--', label='Baseline', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'comparison_speedup.png', dpi=300, bbox_inches='tight')
    print("✓ Generated comparison_speedup.png")
    plt.close()


def plot_scalability():
    """Batched proof time scalability"""
    df = pd.read_csv(METRICS_DIR / 'scalability_batched.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Proving time
    ax1.plot(df['Dataset Size'], df['Proof Time (s)'], marker='o', linewidth=2.5, 
             markersize=8, color='#2ecc71', label='Total Proving Time')
    ax1.fill_between(df['Dataset Size'], 0, df['Proof Time (s)'], alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Proving Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Batched Proving Time Scalability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add time annotations
    for i, row in df.iterrows():
        time_min = row['Proof Time (s)'] / 60
        ax1.annotate(f'{time_min:.1f} min', 
                    xy=(row['Dataset Size'], row['Proof Time (s)']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')
    
    # Verification time
    ax2.bar(df['Dataset Size'], df['Verify Time (ms)'], color='#9b59b6', alpha=0.8, edgecolor='purple')
    ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Verification Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Batched Verification Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'scalability_batched.png', dpi=300, bbox_inches='tight')
    print("✓ Generated scalability_batched.png")
    plt.close()


def plot_proof_size():
    """Proof size comparison"""
    df = pd.read_csv(METRICS_DIR / 'proof_size_batched.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Constraints comparison
    colors = ['#e67e22', '#3498db', '#27ae60']
    bars1 = ax1.barh(df['Circuit Name'], df['Constraints'], color=colors, alpha=0.8)
    ax1.set_xlabel('Number of Constraints', fontsize=12, fontweight='bold')
    ax1.set_title('Circuit Complexity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add constraint labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width):,}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Proof size (constant!)
    bars2 = ax2.bar(df['Circuit Name'], df['Proof Size (Bytes)'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Proof Size (Bytes)', fontsize=12, fontweight='bold')
    ax2.set_title('Proof Size (Constant!)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(df['Circuit Name'], rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=584, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(1, 600, '584 bytes (constant)', ha='center', fontweight='bold', fontsize=11, color='red')
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'proof_size_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated proof_size_comparison.png")
    plt.close()


def plot_throughput():
    """Throughput analysis"""
    df = pd.read_csv(METRICS_DIR / 'proof_time_batched.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate throughput (samples/sec)
    throughput = df['Dataset Size'] / df['Total Time (s)']
    
    ax.plot(df['Dataset Size'], throughput, marker='D', linewidth=2.5, 
            markersize=10, color='#e74c3c', label='Batched Throughput')
    ax.fill_between(df['Dataset Size'], 0, throughput, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
    ax.set_title('Batched ZK-LR Throughput Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Add throughput annotations
    for i, row in df.iterrows():
        tput = row['Dataset Size'] / row['Total Time (s)']
        ax.annotate(f'{tput:.1f} s/s', 
                   xy=(row['Dataset Size'], tput),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'throughput_batched.png', dpi=300, bbox_inches='tight')
    print("✓ Generated throughput_batched.png")
    plt.close()


def plot_batch_analysis():
    """Batch size analysis"""
    df = pd.read_csv(METRICS_DIR / 'scalability_batched.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Number of batches
    ax1.bar(df['Dataset Size'], df['Num Batches'], color='#16a085', alpha=0.8, edgecolor='darkgreen')
    ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Batches', fontsize=12, fontweight='bold')
    ax1.set_title('Batches Required (20 samples/batch)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add batch count labels
    for i, row in df.iterrows():
        ax1.text(row['Dataset Size'], row['Num Batches'] + 2,
                f"{int(row['Num Batches'])}", ha='center', fontweight='bold', fontsize=10)
    
    # Time per batch (constant)
    time_per_batch = df['Proof Time (s)'] / df['Num Batches']
    ax2.plot(df['Dataset Size'], time_per_batch, marker='s', linewidth=2.5,
            markersize=8, color='#e67e22', label='Avg Time/Batch')
    ax2.axhline(y=time_per_batch.mean(), color='blue', linestyle='--', 
               label=f'Mean: {time_per_batch.mean():.1f}s', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time per Batch (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Consistent Batch Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'batch_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Generated batch_analysis.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("BATCHED ZK-LR VISUALIZATION GENERATOR")
    print("="*60 + "\n")
    
    print("Generating PNG visualizations...\n")
    
    plot_comparison()
    plot_scalability()
    plot_proof_size()
    plot_throughput()
    plot_batch_analysis()
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print("="*60)
    print(f"\nSaved to: {METRICS_DIR}\n")
    print("Generated files:")
    print("  📊 comparison_speedup.png      - Individual vs Batched")
    print("  📈 scalability_batched.png     - Time scalability")
    print("  📦 proof_size_comparison.png   - Constraints & proof size")
    print("  ⚡ throughput_batched.png       - Throughput analysis")
    print("  🔢 batch_analysis.png          - Batch performance")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
