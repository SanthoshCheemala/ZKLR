#!/usr/bin/env python3
"""
Visualization Generator for Batch Size Analysis
Generates line graphs showing how performance metrics scale with batch size
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Matplotlib configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / 'metrics_output'


def read_csv(filename):
    """Read CSV file and return headers and data rows"""
    csv_path = METRICS_DIR / filename
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found!")
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    return rows


def plot_proving_time():
    """Plot total proving time vs batch size"""
    rows = read_csv('batch_size_analysis.csv')
    if not rows:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows]
    proving_times = [float(r['Total Prove Time (s)']) for r in rows]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, proving_times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Add value annotations
    for x, y in zip(batch_sizes, proving_times):
        plt.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Batch Size (samples per proof)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Proving Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Proving Time vs Batch Size\n(Fixed Dataset: 500 samples)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = METRICS_DIR / 'batch_proving_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


def plot_throughput():
    """Plot throughput (samples/sec) vs batch size"""
    rows = read_csv('throughput_analysis.csv')
    if not rows:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows]
    throughput = [float(r['Throughput (samples/s)']) for r in rows]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, throughput, marker='s', linewidth=2, markersize=8, color='#A23B72')
    
    # Add value annotations
    for x, y in zip(batch_sizes, throughput):
        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Batch Size (samples per proof)', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
    plt.title('System Throughput vs Batch Size\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = METRICS_DIR / 'batch_throughput.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


def plot_constraints():
    """Plot circuit constraints vs batch size"""
    rows = read_csv('constraint_analysis.csv')
    if not rows:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows]
    total_constraints = [int(r['Total Constraints']) for r in rows]
    constraints_per_sample = [float(r['Constraints/Sample']) for r in rows]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Total constraints
    ax1.plot(batch_sizes, total_constraints, marker='D', linewidth=2, 
            markersize=8, color='#F18F01')
    for x, y in zip(batch_sizes, total_constraints):
        ax1.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Circuit Constraints', fontsize=11, fontweight='bold')
    ax1.set_title('Circuit Complexity Growth', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right: Constraints per sample
    ax2.plot(batch_sizes, constraints_per_sample, marker='v', linewidth=2, 
            markersize=8, color='#6A994E')
    for x, y in zip(batch_sizes, constraints_per_sample):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Constraints/Sample', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Sample Efficiency', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = METRICS_DIR / 'batch_constraints.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


def plot_scalability():
    """Plot proving time per sample to show efficiency"""
    rows = read_csv('batch_size_analysis.csv')
    if not rows:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows]
    total_proving_times = [float(r['Total Prove Time (s)']) for r in rows]
    
    # Calculate proving time per sample (500 samples total)
    proving_time_per_sample = [t / 500 for t in total_proving_times]
    
    # Calculate ideal (constant time per sample - use best observed)
    best_time_per_sample = min(proving_time_per_sample)
    ideal_line = [best_time_per_sample] * len(batch_sizes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Proving time per sample
    ax1.plot(batch_sizes, proving_time_per_sample, marker='o', linewidth=2.5, 
            markersize=8, color='#0077B6', label='Actual')
    ax1.plot(batch_sizes, ideal_line, linestyle='--', linewidth=2, 
            color='#90E0EF', alpha=0.7, label='Optimal (Batch 25)')
    
    # Add value annotations
    for x, y in zip(batch_sizes, proving_time_per_sample):
        ax1.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Proving Time per Sample (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Per-Sample Efficiency', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Efficiency percentage (how close to optimal)
    efficiency = [(best_time_per_sample / t) * 100 for t in proving_time_per_sample]
    colors = ['#d62828' if e < 90 else '#52b788' if e > 95 else '#f77f00' for e in efficiency]
    
    bars = ax2.bar(batch_sizes, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=100, color='#90E0EF', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    
    # Add value annotations on bars
    for bar, e in zip(bars, efficiency):
        height = bar.get_height()
        ax2.annotate(f'{e:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Efficiency Relative to Optimal', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Scalability Analysis: Per-Sample Proving Efficiency', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = METRICS_DIR / 'batch_scalability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


def plot_verification_time():
    """Plot total verification time vs batch size"""
    rows = read_csv('batch_size_analysis.csv')
    if not rows:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows]
    verify_times = [float(r['Total Verify Time (ms)']) / 1000.0 for r in rows]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, verify_times, marker='^', linewidth=2, markersize=8, color='#D62828')
    
    # Add value annotations
    for x, y in zip(batch_sizes, verify_times):
        plt.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Batch Size (samples per proof)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Verification Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Verification Overhead vs Batch Size\n(Fixed Dataset: 500 samples)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = METRICS_DIR / 'batch_verification_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


def plot_comprehensive_comparison():
    """Create a comprehensive 4-panel comparison"""
    rows_batch = read_csv('batch_size_analysis.csv')
    rows_throughput = read_csv('throughput_analysis.csv')
    
    if not rows_batch or not rows_throughput:
        return
    
    batch_sizes = [int(r['Batch Size']) for r in rows_batch]
    proving_times = [float(r['Total Prove Time (s)']) for r in rows_batch]
    constraints = [int(r['Constraints']) for r in rows_batch]
    throughput = [float(r['Throughput (samples/s)']) for r in rows_throughput]
    verify_times = [float(r['Total Verify Time (ms)']) / 1000.0 for r in rows_batch]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Proving time
    ax1.plot(batch_sizes, proving_times, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Proving Time (s)', fontweight='bold')
    ax1.set_title('(A) Total Proving Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Throughput
    ax2.plot(batch_sizes, throughput, marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Throughput (samples/s)', fontweight='bold')
    ax2.set_title('(B) System Throughput', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Constraints
    ax3.plot(batch_sizes, constraints, marker='D', linewidth=2, markersize=6, color='#F18F01')
    ax3.set_xlabel('Batch Size', fontweight='bold')
    ax3.set_ylabel('Total Constraints', fontweight='bold')
    ax3.set_title('(C) Circuit Complexity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Verification time
    ax4.plot(batch_sizes, verify_times, marker='^', linewidth=2, markersize=6, color='#D62828')
    ax4.set_xlabel('Batch Size', fontweight='bold')
    ax4.set_ylabel('Verification Time (s)', fontweight='bold')
    ax4.set_title('(D) Total Verification Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Batch Size Analysis - Comprehensive Comparison\n(Fixed Dataset: 500 samples)', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = METRICS_DIR / 'batch_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created {output_path}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BATCH ANALYSIS VISUALIZATION GENERATOR")
    print("="*70)
    
    plot_proving_time()
    plot_throughput()
    plot_constraints()
    plot_scalability()
    plot_verification_time()
    plot_comprehensive_comparison()
    
    print("\n✓ All visualizations generated successfully!")
    print("="*70)
