#!/usr/bin/env python3
"""
Metrics Collector for Batch Size Analysis
- Evaluates performance metrics as batch size varies from 1 to 100
- Fixed dataset size: 500 samples  
- Measures: proving time, verification time, throughput, constraints
- Generates comprehensive CSV files and visualizations
"""

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / 'metrics_output'
DATA_DIR = ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Fixed dataset size for consistent comparison
FIXED_DATASET_SIZE = 500

# Batch sizes to test  
BATCH_SIZES = [5, 10, 20, 25, 50, 100]


def run_go_with_batch_size(batch_size: int):
    """
    Run the batched ZK proof system with specified batch size
    Modifies main_batched.go to use the specified batch size
    Returns metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"🔧 Running with BATCH_SIZE={batch_size}, DATASET_SIZE={FIXED_DATASET_SIZE}")
    print(f"[VERBOSE] Total samples: {FIXED_DATASET_SIZE}")
    print(f"[VERBOSE] Expected batches: {FIXED_DATASET_SIZE // batch_size}")
    print(f"[VERBOSE] Samples per batch: {batch_size}")
    print(f"{'='*70}")
    
    # Modify main_batched.go to use the specified batch size
    go_file_path = ROOT / 'main_batched.go'
    backup_path = ROOT / 'main_batched.go.backup'
    
    # Backup original file (first time only)
    if not backup_path.exists():
        print(f"[VERBOSE] Creating backup: {backup_path}")
        with open(go_file_path, 'r') as f:
            original_code = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_code)
        print(f"[VERBOSE] ✓ Backup created")
    else:
        print(f"[VERBOSE] Using existing backup")
    
    # Read current file
    print(f"[VERBOSE] Modifying BatchSize to {batch_size}")
    with open(go_file_path, 'r') as f:
        go_code = f.read()
    
    # Replace BatchSize constant
    go_code_modified = re.sub(
        r'const BatchSize = \d+',
        f'const BatchSize = {batch_size}',
        go_code
    )
    
    # Also replace testSize to use our fixed dataset size
    go_code_modified = re.sub(
        r'testSize := \d+',
        f'testSize := {FIXED_DATASET_SIZE}',
        go_code_modified
    )
    
    # Write modified file
    with open(go_file_path, 'w') as f:
        f.write(go_code_modified)
    
    # Build command
    cmd = ['go', 'run', 'main_batched.go']
    
    print(f"[VERBOSE] Starting execution at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"[VERBOSE] ✓ Completed in {elapsed:.2f}s at {time.strftime('%H:%M:%S')}")
        
        if result.returncode != 0:
            print(f"ERROR: Go program failed with code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None
        
        # Print output
        print(result.stdout)
        
        # Parse metrics from cache file
        metrics_file = CACHE_DIR / 'run_metrics_batched.txt'
        if not metrics_file.exists():
            print(f"ERROR: Metrics file not found: {metrics_file}")
            return None
        
        metrics = {}
        with open(metrics_file) as f:
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=', 1)
                    try:
                        metrics[key] = float(val)
                    except:
                        metrics[key] = val
        
        metrics['total_wall_time_s'] = elapsed
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout after 10 minutes for batch_size={batch_size}")
        return None
    except Exception as e:
        print(f"ERROR: Exception running batch_size={batch_size}: {e}")
        return None


def collect_all_metrics():
    """Run experiments for all batch sizes"""
    print(f"\n{'='*70}")
    print(f"🚀 STARTING BATCH ANALYSIS")
    print(f"{'='*70}")
    print(f"[VERBOSE] Batch sizes: {BATCH_SIZES}")
    print(f"[VERBOSE] Dataset size: {FIXED_DATASET_SIZE}")
    print(f"[VERBOSE] Total tests: {len(BATCH_SIZES)}")
    
    results = []
    overall_start = time.time()
    
    for idx, batch_size in enumerate(BATCH_SIZES, 1):
        print(f"\n{'='*70}")
        print(f"📊 TEST {idx}/{len(BATCH_SIZES)}: Batch Size = {batch_size}")
        print(f"{'='*70}")
        metrics = run_go_with_batch_size(batch_size)
        if metrics:
            results.append({
                'batch_size': batch_size,
                'metrics': metrics
            })
            print(f"[VERBOSE] ✅ Test {idx}/{len(BATCH_SIZES)} completed")
        else:
            print(f"[VERBOSE] ⚠️  Skipping batch_size={batch_size} due to errors")
    
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"✅ ALL TESTS COMPLETED")
    print(f"{'='*70}")
    print(f"[VERBOSE] Total time: {overall_elapsed/60:.1f} minutes")
    print(f"[VERBOSE] Successful: {len(results)}/{len(BATCH_SIZES)}")
    
    # Restore original file after all tests
    print(f"\n[VERBOSE] Restoring original main_batched.go")
    go_file_path = ROOT / 'main_batched.go'
    backup_path = ROOT / 'main_batched.go.backup'
    if backup_path.exists():
        with open(backup_path, 'r') as f:
            original_code = f.read()
        with open(go_file_path, 'w') as f:
            f.write(original_code)
        print(f"[VERBOSE] ✓ Restored original file")
    
    return results


def write_batch_analysis_csv(results):
    """Generate comprehensive batch analysis CSV"""
    rows = []
    
    for r in results:
        batch_size = r['batch_size']
        m = r['metrics']
        
        num_batches = int(m.get('num_batches', 0))
        constraints = int(m.get('constraints', 0))
        setup_time_s = m.get('setup_time_ms', 0) / 1000.0
        total_prove_time_s = m.get('total_proving_time_ms', 0) / 1000.0
        avg_prove_time_s = m.get('avg_prove_time_ms', 0) / 1000.0
        verify_time_ms = m.get('avg_verify_time_ms', 0)
        total_verify_time_ms = num_batches * verify_time_ms
        proof_size_bytes = int(m.get('proof_size_bytes', 0))
        proof_size_kb = m.get('proof_size_kb', 0)
        throughput = m.get('throughput_samples_per_sec', 0)
        accuracy = m.get('accuracy', 0)
        
        rows.append([
            batch_size,
            FIXED_DATASET_SIZE,
            num_batches,
            constraints,
            f"{setup_time_s:.2f}",
            f"{total_prove_time_s:.2f}",
            f"{avg_prove_time_s:.3f}",
            f"{verify_time_ms:.2f}",
            f"{total_verify_time_ms:.2f}",
            proof_size_bytes,
            f"{proof_size_kb:.2f}",
            f"{throughput:.2f}",
            f"{accuracy:.2f}"
        ])
    
    csv_path = METRICS_DIR / 'batch_size_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Batch Size',
            'Dataset Size',
            'Num Batches',
            'Constraints',
            'Setup Time (s)',
            'Total Prove Time (s)',
            'Avg Prove Time/Batch (s)',
            'Avg Verify Time/Batch (ms)',
            'Total Verify Time (ms)',
            'Proof Size (bytes)',
            'Proof Size (KB)',
            'Throughput (samples/s)',
            'Accuracy (%)'
        ])
        writer.writerows(rows)
    
    print(f"\n✓ Created {csv_path}")


def write_throughput_analysis_csv(results):
    """
    Throughput Analysis: successful verifications per unit time
    Throughput = (Number of verified proofs) / (Total time)
    """
    rows = []
    
    for r in results:
        batch_size = r['batch_size']
        m = r['metrics']
        
        num_batches = int(m.get('num_batches', 0))
        total_prove_time_s = m.get('total_proving_time_ms', 0) / 1000.0
        verify_time_ms = m.get('avg_verify_time_ms', 0)
        total_verify_time_s = (num_batches * verify_time_ms) / 1000.0
        
        # Total end-to-end time (prove + verify)
        total_time_s = total_prove_time_s + total_verify_time_s
        
        # Throughput = verified proofs / time
        if total_time_s > 0:
            throughput_proofs_per_sec = num_batches / total_time_s
            throughput_samples_per_sec = FIXED_DATASET_SIZE / total_time_s
        else:
            throughput_proofs_per_sec = 0
            throughput_samples_per_sec = 0
        
        rows.append([
            batch_size,
            num_batches,
            FIXED_DATASET_SIZE,
            f"{total_prove_time_s:.2f}",
            f"{total_verify_time_s:.3f}",
            f"{total_time_s:.2f}",
            f"{throughput_proofs_per_sec:.2f}",
            f"{throughput_samples_per_sec:.2f}"
        ])
    
    csv_path = METRICS_DIR / 'throughput_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Batch Size',
            'Total Proofs Generated',
            'Total Samples Verified',
            'Total Proving Time (s)',
            'Total Verification Time (s)',
            'Total Time (s)',
            'Throughput (proofs/s)',
            'Throughput (samples/s)'
        ])
        writer.writerows(rows)
    
    print(f"✓ Created {csv_path}")


def write_constraint_analysis_csv(results):
    """Analyze constraint growth with batch size"""
    rows = []
    
    for r in results:
        batch_size = r['batch_size']
        m = r['metrics']
        
        constraints = int(m.get('constraints', 0))
        constraints_per_sample = constraints / batch_size if batch_size > 0 else 0
        
        rows.append([
            batch_size,
            constraints,
            f"{constraints_per_sample:.0f}"
        ])
    
    csv_path = METRICS_DIR / 'constraint_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Batch Size',
            'Total Constraints',
            'Constraints/Sample'
        ])
        writer.writerows(rows)
    
    print(f"✓ Created {csv_path}")


def write_scalability_characteristics_csv(results):
    """
    Scalability Characteristics: How performance scales with batch size
    """
    rows = []
    
    # Use batch_size=1 as baseline
    baseline = None
    for r in results:
        if r['batch_size'] == 1:
            baseline = r
            break
    
    if not baseline:
        print("⚠ Warning: No batch_size=1 baseline found")
        baseline = results[0]  # Use first result as baseline
    
    baseline_time = baseline['metrics'].get('total_proving_time_ms', 1) / 1000.0
    
    for r in results:
        batch_size = r['batch_size']
        m = r['metrics']
        
        total_prove_time_s = m.get('total_proving_time_ms', 0) / 1000.0
        speedup = baseline_time / total_prove_time_s if total_prove_time_s > 0 else 0
        
        num_batches = int(m.get('num_batches', 0))
        efficiency = (speedup / batch_size) * 100 if batch_size > 0 else 0
        
        rows.append([
            batch_size,
            num_batches,
            f"{total_prove_time_s:.2f}",
            f"{speedup:.2f}",
            f"{efficiency:.1f}"
        ])
    
    csv_path = METRICS_DIR / 'scalability_characteristics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Batch Size',
            'Num Batches',
            'Total Proving Time (s)',
            'Speedup vs Batch=1',
            'Efficiency (%)'
        ])
        writer.writerows(rows)
    
    print(f"✓ Created {csv_path}")


def save_full_results_json(results):
    """Save complete results to JSON for further analysis"""
    output = {
        'dataset_size': FIXED_DATASET_SIZE,
        'batch_sizes_tested': BATCH_SIZES,
        'results': []
    }
    
    for r in results:
        output['results'].append({
            'batch_size': r['batch_size'],
            'metrics': r['metrics']
        })
    
    json_path = METRICS_DIR / 'batch_analysis_full.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Created {json_path}")


def print_summary(results):
    """Print summary table"""
    print("\n" + "="*80)
    print(f"BATCH SIZE ANALYSIS SUMMARY (Dataset Size: {FIXED_DATASET_SIZE} samples)")
    print("="*80)
    print(f"{'Batch':<8} {'Batches':<8} {'Constraints':<12} {'Prove(s)':<10} {'Verify(ms)':<11} {'Throughput':<15}")
    print(f"{'Size':<8} {'Count':<8} {'':<12} {'Total':<10} {'Total':<11} {'(samples/s)':<15}")
    print("-"*80)
    
    for r in results:
        batch_size = r['batch_size']
        m = r['metrics']
        
        num_batches = int(m.get('num_batches', 0))
        constraints = int(m.get('constraints', 0))
        prove_time = m.get('total_proving_time_ms', 0) / 1000.0
        verify_time = num_batches * m.get('avg_verify_time_ms', 0)
        throughput = m.get('throughput_samples_per_sec', 0)
        
        print(f"{batch_size:<8} {num_batches:<8} {constraints:<12,} {prove_time:<10.2f} {verify_time:<11.1f} {throughput:<15.2f}")
    
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("ZKLR BATCH SIZE ANALYSIS")
    print(f"Fixed Dataset Size: {FIXED_DATASET_SIZE} samples")
    print(f"Batch Sizes to Test: {BATCH_SIZES}")
    print("="*80)
    
    # Collect metrics for all batch sizes
    results = collect_all_metrics()
    
    if not results:
        print("\n❌ ERROR: No results collected. Exiting.")
        sys.exit(1)
    
    print(f"\n✓ Successfully collected metrics for {len(results)} batch sizes")
    
    # Generate all CSV files
    print("\nGenerating CSV files...")
    write_batch_analysis_csv(results)
    write_throughput_analysis_csv(results)
    write_constraint_analysis_csv(results)
    write_scalability_characteristics_csv(results)
    save_full_results_json(results)
    
    # Print summary
    print_summary(results)
    
    print("✓ All metrics collected and saved successfully!")
    print(f"\nOutput directory: {METRICS_DIR}")
    print("\nGenerated files:")
    print("  - batch_size_analysis.csv        (comprehensive metrics)")
    print("  - throughput_analysis.csv        (successful verifications/time)")
    print("  - constraint_analysis.csv        (circuit complexity)")
    print("  - scalability_characteristics.csv (performance scaling)")
    print("  - batch_analysis_full.json       (complete raw data)")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review CSV files for insights")
    print("  2. Generate visualizations: python3 metrics_output/plot_batch_analysis.py")
    print("  3. Analyze throughput and scalability trends")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
