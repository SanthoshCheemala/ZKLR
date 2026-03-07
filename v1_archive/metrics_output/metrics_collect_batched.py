#!/usr/bin/env python3
"""
Metrics Collector for ZKLR BATCHED VERSION
- Reads metrics from batched proof run
- Generates performance CSVs and plots
"""

import csv
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / 'metrics_output'
RUN_METRICS_BATCHED = ROOT / 'data' / 'cache' / 'run_metrics_batched.txt'

def parse_run_metrics():
    """Parse the exported metrics from data/cache/run_metrics_batched.txt"""
    if not RUN_METRICS_BATCHED.exists():
        print(f"ERROR: Metrics file not found: {RUN_METRICS_BATCHED}")
        print("Please run ./zkproof_batched first!")
        return None
    
    metrics = {}
    with open(RUN_METRICS_BATCHED) as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                try:
                    metrics[key] = float(val)
                except:
                    metrics[key] = val
    return metrics


def write_proof_size_csv(metrics):
    rows = []
    kb = metrics.get('proof_size_kb', 0.0)
    b = int(metrics.get('proof_size_bytes', 0))
    constraints = int(metrics.get('constraints', 0))
    batch_size = int(metrics.get('batch_size', 20))
    
    rows.append(['Linear Circuit (per sample)', 3, f"{kb:.2f}", b])
    rows.append(['Sigmoid LUT (per sample)', 58019, f"{kb:.2f}", b])
    rows.append([f'Batched Circuit ({batch_size} samples)', constraints, f"{kb:.2f}", b])

    with open(METRICS_DIR / 'proof_size_batched.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit Name','Constraints','Proof Size (KB)','Proof Size (Bytes)'])
        writer.writerows(rows)
    print(f"✓ Created {METRICS_DIR / 'proof_size_batched.csv'}")


def write_proof_time_csv(metrics):
    samples = int(metrics.get('samples', 100))
    batch_size = int(metrics.get('batch_size', 20))
    num_batches = int(metrics.get('num_batches', 5))
    avg_prove_ms = metrics.get('avg_prove_time_ms', 8000)
    throughput_samples = metrics.get('throughput_samples_per_sec', 2.0)
    
    entries = []
    for size in [100, 500, 1000, 2000, 3000]:
        batches = (size + batch_size - 1) // batch_size
        total_time_s = size / throughput_samples
        entries.append({
            'Dataset Size': size,
            'Batches': batches,
            'Samples/Batch': batch_size,
            'Time (s)': round(total_time_s, 1),
        })
    
    with open(METRICS_DIR / 'proof_time_batched.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset Size','Num Batches','Samples/Batch','Total Time (s)'])
        for e in entries:
            writer.writerow([e['Dataset Size'], e['Batches'], e['Samples/Batch'], e['Time (s)']])
    print(f"✓ Created {METRICS_DIR / 'proof_time_batched.csv'}")


def write_comparison_csv(metrics):
    """Compare individual vs batched approach"""
    batch_size = int(metrics.get('batch_size', 20))
    throughput_batched = metrics.get('throughput_samples_per_sec', 2.0)
    
    # Individual approach: ~1.6 samples/sec (from previous runs)
    throughput_individual = 1.6
    
    entries = []
    for size in [100, 500, 1000, 2000, 3000]:
        time_individual = size / throughput_individual
        time_batched = size / throughput_batched
        speedup = time_individual / time_batched
        
        entries.append({
            'Samples': size,
            'Individual Time (s)': round(time_individual, 1),
            'Batched Time (s)': round(time_batched, 1),
            'Speedup': f"{speedup:.1f}x",
        })
    
    with open(METRICS_DIR / 'comparison_batched.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset Size','Individual Time (s)','Batched Time (s)','Speedup'])
        for e in entries:
            writer.writerow([e['Samples'], e['Individual Time (s)'], 
                           e['Batched Time (s)'], e['Speedup']])
    print(f"✓ Created {METRICS_DIR / 'comparison_batched.csv'}")


def write_scalability_csv(metrics):
    batch_size = int(metrics.get('batch_size', 20))
    throughput = metrics.get('throughput_samples_per_sec', 2.0)
    verify_ms = metrics.get('avg_verify_time_ms', 2.0)
    
    entries = []
    for size in [100, 500, 1000, 2000, 3000]:
        batches = (size + batch_size - 1) // batch_size
        prove_time = size / throughput
        total_verify = batches * verify_ms
        
        entries.append({
            'Dataset Size': size,
            'Num Batches': batches,
            'Samples/Batch': batch_size,
            'Proof Time (s)': round(prove_time, 1),
            'Verify Time (ms)': round(total_verify, 1),
        })
    
    with open(METRICS_DIR / 'scalability_batched.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset Size','Num Batches','Samples/Batch','Proof Time (s)','Verify Time (ms)'])
        for e in entries:
            writer.writerow([e['Dataset Size'], e['Num Batches'], e['Samples/Batch'],
                           e['Proof Time (s)'], e['Verify Time (ms)']])
    print(f"✓ Created {METRICS_DIR / 'scalability_batched.csv'}")


def update_metrics_json(metrics):
    payload = {
        'batched': True,
        'batch_size': int(metrics.get('batch_size', 20)),
        'num_batches': int(metrics.get('num_batches', 0)),
        'proof': {
            'bytes': int(metrics.get('proof_size_bytes', 0)),
            'kb': metrics.get('proof_size_kb', 0.0),
            'verify_ms': metrics.get('avg_verify_time_ms', 0.0)
        },
        'performance': {
            'setup_time_ms': metrics.get('setup_time_ms', 0.0),
            'total_proving_time_ms': metrics.get('total_proving_time_ms', 0.0),
            'avg_prove_time_per_batch_ms': metrics.get('avg_prove_time_ms', 0.0),
            'samples': int(metrics.get('samples', 0)),
            'accuracy': metrics.get('accuracy', 0.0),
            'constraints': int(metrics.get('constraints', 0)),
            'throughput_samples_per_sec': metrics.get('throughput_samples_per_sec', 0.0),
            'throughput_batches_per_sec': metrics.get('throughput_batches_per_sec', 0.0),
        }
    }
    with open(METRICS_DIR / 'metrics_batched.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Created {METRICS_DIR / 'metrics_batched.json'}")


def print_summary(metrics):
    print("\n" + "="*60)
    print("BATCHED METRICS SUMMARY")
    print("="*60)
    print(f"Batch Size:           {int(metrics.get('batch_size', 0))} samples/batch")
    print(f"Total Batches:        {int(metrics.get('num_batches', 0))}")
    print(f"Total Samples:        {int(metrics.get('samples', 0))}")
    print(f"Accuracy:             {metrics.get('accuracy', 0.0):.2f}%")
    print(f"Constraints/Batch:    {int(metrics.get('constraints', 0)):,}")
    print(f"Proof Size:           {int(metrics.get('proof_size_bytes', 0))} bytes")
    print(f"Setup Time:           {metrics.get('setup_time_ms', 0.0)/1000:.1f}s")
    print(f"Total Proving Time:   {metrics.get('total_proving_time_ms', 0.0)/1000:.1f}s")
    print(f"Avg Time/Batch:       {metrics.get('avg_prove_time_ms', 0.0)/1000:.1f}s")
    print(f"Throughput:           {metrics.get('throughput_samples_per_sec', 0.0):.2f} samples/sec")
    print("="*60 + "\n")


def main():
    print("ZKLR Batched Metrics Collector")
    print("="*60 + "\n")
    
    metrics = parse_run_metrics()
    if not metrics:
        sys.exit(1)
    
    print("Generating CSV files...")
    write_proof_size_csv(metrics)
    write_proof_time_csv(metrics)
    write_comparison_csv(metrics)
    write_scalability_csv(metrics)
    update_metrics_json(metrics)
    
    print_summary(metrics)
    
    print("✓ All metrics collected successfully!")
    print(f"\nMetrics saved to: {METRICS_DIR}")
    print("\nGenerated files:")
    print("  - proof_size_batched.csv")
    print("  - proof_time_batched.csv")
    print("  - comparison_batched.csv")
    print("  - scalability_batched.csv")
    print("  - metrics_batched.json")
    
    # Generate PNG visualizations
    print("\n" + "="*60)
    print("Generating PNG visualizations...")
    print("="*60 + "\n")
    
    plot_script = METRICS_DIR / 'plot_metrics_batched.py'
    if plot_script.exists():
        try:
            subprocess.run(['python3', str(plot_script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Visualization generation failed: {e}")
            print("  Install dependencies: pip3 install matplotlib pandas numpy")
    else:
        print(f"⚠ Warning: Plot script not found: {plot_script}")


if __name__ == '__main__':
    main()
