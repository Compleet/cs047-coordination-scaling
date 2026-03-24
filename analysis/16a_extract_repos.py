#!/usr/bin/env python3
"""
CS-047 CSD Robustness Analysis — Phase 1: Extract Per-Repo Summary
===================================================================

Streams all GHTorrent PR comment CSVs and produces repo_summary.csv
with one row per qualified repo (N >= 5, comments >= 20, PRs >= 5).

Usage:
    python 16a_extract_repos.py /path/to/ghtorrent/csvs

Output:
    results/csd_robustness/repo_summary.csv

Expected runtime: 2-3 hours on ~90GB, ~440M rows.
Memory: ~2GB (streaming, no full dataset in memory).
"""

import csv
import glob
import os
import sys
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "csd_robustness"


def stream_all_files(directory):
    """Stream all GHTorrent CSV files and collect per-repo statistics."""
    files = sorted(glob.glob(os.path.join(directory, "ghtorrent-*.csv")))
    if not files:
        # Try alternative patterns
        files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    
    print(f"Found {len(files)} files to process")
    if not files:
        print("ERROR: No CSV files found. Check the directory path.")
        sys.exit(1)

    repo_stats = defaultdict(lambda: {
        'contributors': set(),
        'comments': 0,
        'prs': set(),
    })

    total_rows = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            file_rows = 0

            for row in reader:
                file_rows += 1
                if file_rows % 10_000_000 == 0:
                    print(f"  {file_rows:,} rows... ({len(repo_stats):,} repos so far)")

                repo = row.get('repo', '')
                actor = row.get('actor_login', '')
                author = row.get('author_login', '')
                pr_id = row.get('pr_id', '')

                if not repo:
                    continue

                stats = repo_stats[repo]
                if actor:
                    stats['contributors'].add(actor)
                if author:
                    stats['contributors'].add(author)
                stats['comments'] += 1
                if pr_id:
                    stats['prs'].add(pr_id)

            total_rows += file_rows
            print(f"  Done: {file_rows:,} rows")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_rows:,} rows across {len(repo_stats):,} repos")
    print(f"{'='*60}")

    return repo_stats, total_rows


def filter_and_save(repo_stats, total_rows, min_n=5, min_activity=20, min_prs=5):
    """Filter repos and save to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = RESULTS_DIR / "repo_summary.csv"
    
    n_qualified = 0
    n_total = len(repo_stats)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'repo', 'n_contributors', 'n_prs', 'n_comments',
            'comments_per_pr', 'prs_per_person', 'log_n', 'log_prs'
        ])
        
        for repo, stats in repo_stats.items():
            n = len(stats['contributors'])
            comments = stats['comments']
            prs = len(stats['prs'])
            
            if n >= min_n and comments >= min_activity and prs >= min_prs:
                cpp = comments / prs if prs > 0 else 0
                ppp = prs / n if n > 0 else 0
                log_n = np.log(n) if n > 0 else 0
                log_prs = np.log(prs) if prs > 0 else 0
                
                writer.writerow([
                    repo, n, prs, comments, 
                    f"{cpp:.4f}", f"{ppp:.4f}",
                    f"{log_n:.6f}", f"{log_prs:.6f}"
                ])
                n_qualified += 1
    
    # Save metadata
    meta = {
        'total_rows_processed': total_rows,
        'total_repos': n_total,
        'qualified_repos': n_qualified,
        'filters': {
            'min_contributors': min_n,
            'min_comments': min_activity,
            'min_prs': min_prs,
        }
    }
    
    with open(RESULTS_DIR / "extraction_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nSaved {n_qualified:,} qualified repos to {output_path}")
    print(f"Filtered from {n_total:,} total repos")
    print(f"  (min_n={min_n}, min_comments={min_activity}, min_prs={min_prs})")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python 16a_extract_repos.py /path/to/ghtorrent/csvs")
        print("\nExpects directory containing ghtorrent-*.csv files")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Phase 1: Extract Per-Repo Summary from GHTorrent")
    print("=" * 60)
    
    repo_stats, total_rows = stream_all_files(data_dir)
    output_path = filter_and_save(repo_stats, total_rows)
    
    # Quick preview
    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"\nPreview of repo_summary.csv:")
    print(f"  Shape: {df.shape}")
    print(f"  N range: {df['n_contributors'].min()} – {df['n_contributors'].max()}")
    print(f"  Median N: {df['n_contributors'].median():.0f}")
    print(f"\n  Distribution by size:")
    bins = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
    for i in range(len(bins)-1):
        count = ((df['n_contributors'] >= bins[i]) & (df['n_contributors'] < bins[i+1])).sum()
        print(f"    N={bins[i]:>5}–{bins[i+1]:<5}: {count:>6,} repos")
    count = (df['n_contributors'] >= bins[-1]).sum()
    print(f"    N≥{bins[-1]:<5}      : {count:>6,} repos")
    
    print(f"\nPhase 1 complete. Now run: python 16b_csd_robustness.py")


if __name__ == '__main__':
    main()
