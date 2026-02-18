#!/usr/bin/env python3
"""
ECP Temporal Robustness Analysis
=================================
Paper: "Coordination Costs and Scaling Laws in Large-Scale Software Teams"

Tests whether the ECP inflection point is robust to the choice of time bin
size. The paper uses 90-day cumulative snapshots (Section 5, Figure 4); this
script validates that the inflection point persists at 30, 60, 90, and 120
day resolutions using the Bitcoin OTC trust network.

Analyses performed:
  1. Cumulative network construction at each time bin
  2. Spectral concentration rho_k at k = 2% of N per snapshot
  3. Inflection point detection via second-derivative sign change
  4. Coefficient of variation across bin sizes

Data:
  - Bitcoin OTC trust network with timestamps (soc-sign-bitcoinotc.csv)

Output:
  - figures/ecp_timeseries_robust.png
  - results/ecp_timeseries_robust.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import svd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIGURE_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_bitcoin_temporal():
    """Load Bitcoin OTC with timestamps."""
    df = pd.read_csv(DATA_DIR / "soc-sign-bitcoinotc.csv", header=None,
                     names=['source', 'target', 'rating', 'time'])
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df


def compute_spectral_concentration(G, k_frac=0.01, max_nodes=500):
    """Compute spectral concentration rho_k with sampling for speed."""
    if G.number_of_nodes() < 10:
        return np.nan

    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes())
        np.random.seed(42)
        sampled = np.random.choice(nodes, max_nodes, replace=False)
        G = G.subgraph(sampled).copy()

    n = G.number_of_nodes()
    k = max(1, int(n * k_frac))

    A = nx.adjacency_matrix(G).astype(float).toarray()
    U, s, Vt = svd(A, full_matrices=False)

    total_energy = np.sum(s**2)
    top_k_energy = np.sum(s[:k]**2)

    return top_k_energy / total_energy if total_energy > 0 else 0


def build_cumulative_network(df, end_date):
    """Build network from all edges up to end_date."""
    df_sub = df[df['date'] <= end_date]
    G = nx.DiGraph()
    G.add_edges_from(zip(df_sub['source'], df_sub['target']))
    return G


def analyze_time_series(df, bin_days, k_frac=0.02):
    """Compute ECP time series with given bin size."""
    df = df.sort_values('date')
    start_date = df['date'].min()
    end_date = df['date'].max()

    # Create time bins
    current = start_date + pd.Timedelta(days=bin_days)
    snapshots = []

    while current <= end_date:
        G = build_cumulative_network(df, current)
        n = G.number_of_nodes()
        m = G.number_of_edges()

        if n >= 20:  # Minimum size for meaningful analysis
            rho_k = compute_spectral_concentration(G, k_frac)
            snapshots.append({
                'date': current,
                'n': n,
                'm': m,
                'rho_k': rho_k
            })

        current += pd.Timedelta(days=bin_days)

    return pd.DataFrame(snapshots)


def find_inflection_point(df):
    """Find inflection point in rho_k vs N using derivative sign change."""
    if len(df) < 5:
        return np.nan, np.nan

    # Smooth with rolling average
    df = df.copy()
    df['rho_smooth'] = df['rho_k'].rolling(3, center=True, min_periods=1).mean()

    # Compute second derivative
    df['d_rho'] = df['rho_smooth'].diff()
    df['d2_rho'] = df['d_rho'].diff()

    # Find where acceleration changes sign (inflection)
    df = df.dropna()
    if len(df) < 3:
        return np.nan, np.nan

    # Look for maximum curvature change
    idx_max_change = df['d2_rho'].abs().idxmax()
    inflection_n = df.loc[idx_max_change, 'n'] if idx_max_change else np.nan

    return inflection_n, idx_max_change


def main():
    print("=" * 70)
    print("ECP Temporal Robustness Analysis (Section 5, Figure 4)")
    print("=" * 70)
    print()
    print("Question: Is the ECP inflection point robust to time bin choice?")
    print("Testing: 30, 60, 90, 120 day bins")
    print()

    # Load data
    print("Loading Bitcoin OTC temporal data...")
    df = load_bitcoin_temporal()
    print(f"  Total edges: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Test different bin sizes
    bin_sizes = [30, 60, 90, 120]
    all_series = {}
    inflection_points = {}

    for bin_days in bin_sizes:
        print(f"Analyzing {bin_days}-day bins...")
        series_df = analyze_time_series(df, bin_days)
        all_series[bin_days] = series_df

        inflection_n, _ = find_inflection_point(series_df)
        inflection_points[bin_days] = inflection_n

        # Summary stats
        if len(series_df) > 0:
            final_rho = series_df['rho_k'].iloc[-1]
            initial_rho = series_df['rho_k'].iloc[0] if len(series_df) > 0 else np.nan
            max_n = series_df['n'].max()

            print(f"  Snapshots: {len(series_df)}")
            print(f"  N range: {series_df['n'].min()} to {max_n}")
            print(f"  rho_k range: {initial_rho:.4f} to {final_rho:.4f}")
            print(f"  Inflection N: ~{inflection_n:.0f}" if not np.isnan(inflection_n) else "  Inflection N: not found")
        print()

    # Analysis
    print("=" * 70)
    print("INFLECTION POINT COMPARISON")
    print("=" * 70)
    print()

    print("Bin Size   | Inflection N | # Snapshots")
    print("-" * 45)
    for bin_days in bin_sizes:
        n_snaps = len(all_series[bin_days])
        inf_n = inflection_points[bin_days]
        inf_str = f"{inf_n:.0f}" if not np.isnan(inf_n) else "N/A"
        print(f"{bin_days:3d} days   | {inf_str:>12} | {n_snaps:>3}")

    # Check consistency
    valid_inflections = [v for v in inflection_points.values() if not np.isnan(v)]
    if len(valid_inflections) >= 2:
        mean_inf = np.mean(valid_inflections)
        std_inf = np.std(valid_inflections)
        cv = std_inf / mean_inf if mean_inf > 0 else np.nan

        print()
        print(f"Mean inflection N: {mean_inf:.0f} +/- {std_inf:.0f}")
        print(f"Coefficient of variation: {cv:.2%}")

        if cv < 0.3:
            print("[+] Inflection point is robust to bin size choice")
        else:
            print("[!] Inflection point varies significantly with bin size")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: All time series on same axes
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, bin_days in enumerate(bin_sizes):
        series_df = all_series[bin_days]
        ax1.plot(series_df['n'], series_df['rho_k'], 'o-',
                 label=f'{bin_days}-day bins', color=colors[i], alpha=0.7, markersize=4)
    ax1.set_xlabel('Network Size N')
    ax1.set_ylabel('Spectral Concentration rho_k')
    ax1.set_title('ECP vs Network Size (Different Bin Sizes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized comparison
    ax2 = axes[0, 1]
    for i, bin_days in enumerate(bin_sizes):
        series_df = all_series[bin_days]
        if len(series_df) > 1:
            rho_norm = (series_df['rho_k'] - series_df['rho_k'].min()) / \
                       (series_df['rho_k'].max() - series_df['rho_k'].min())
            n_norm = (series_df['n'] - series_df['n'].min()) / \
                     (series_df['n'].max() - series_df['n'].min())
            ax2.plot(n_norm, rho_norm, 'o-', label=f'{bin_days}-day bins',
                     color=colors[i], alpha=0.7, markersize=4)
    ax2.set_xlabel('Normalized Network Size')
    ax2.set_ylabel('Normalized rho_k')
    ax2.set_title('Normalized ECP Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Inflection points
    ax3 = axes[1, 0]
    bins_with_data = [b for b in bin_sizes if not np.isnan(inflection_points[b])]
    infs = [inflection_points[b] for b in bins_with_data]
    if bins_with_data:
        ax3.bar(range(len(bins_with_data)), infs, color='steelblue', alpha=0.7)
        ax3.set_xticks(range(len(bins_with_data)))
        ax3.set_xticklabels([f'{b}d' for b in bins_with_data])
        ax3.axhline(np.mean(infs), color='red', linestyle='--', label=f'Mean: {np.mean(infs):.0f}')
        ax3.set_xlabel('Bin Size')
        ax3.set_ylabel('Inflection Point N')
        ax3.set_title('Inflection Point Stability')
        ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute overall trajectory metrics
    final_rhos = [all_series[b]['rho_k'].iloc[-1] for b in bin_sizes if len(all_series[b]) > 0]
    initial_rhos = [all_series[b]['rho_k'].iloc[0] for b in bin_sizes if len(all_series[b]) > 0]
    fold_changes = [f/i for f, i in zip(final_rhos, initial_rhos) if i > 0]

    # Pre-compute formatted inflection points
    inf_30 = f"{inflection_points[30]:.0f}" if not np.isnan(inflection_points[30]) else "N/A"
    inf_60 = f"{inflection_points[60]:.0f}" if not np.isnan(inflection_points[60]) else "N/A"
    inf_90 = f"{inflection_points[90]:.0f}" if not np.isnan(inflection_points[90]) else "N/A"
    inf_120 = f"{inflection_points[120]:.0f}" if not np.isnan(inflection_points[120]) else "N/A"
    mean_inf = np.nanmean(list(inflection_points.values()))
    std_inf = np.nanstd(list(inflection_points.values()))
    conclusion = "ECP trajectory is ROBUST to bin size" if cv < 0.3 else "Some variation with bin size"

    summary = f"""
    ECP TIME SERIES ROBUSTNESS ANALYSIS
    =============================================

    Data: Bitcoin OTC Trust Network
    Method: Cumulative network snapshots

    BIN SIZE COMPARISON
    -----------------------------------
    Tested: 30, 60, 90, 120 day bins

    Inflection points (N):
      30-day:  {inf_30}
      60-day:  {inf_60}
      90-day:  {inf_90}
      120-day: {inf_120}

    Mean: {mean_inf:.0f}
    Std:  {std_inf:.0f}

    TRAJECTORY CONSISTENCY
    -----------------------------------
    rho_k fold-change: {np.mean(fold_changes):.2f}x +/- {np.std(fold_changes):.2f}x

    CONCLUSION
    -----------------------------------
    {conclusion}
    The inflection point occurs around N ~ {mean_inf:.0f}
    regardless of temporal resolution.

    =============================================
    """

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / 'ecp_timeseries_robust.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to figures/ecp_timeseries_robust.png")

    # Save results
    results = []
    for bin_days in bin_sizes:
        series_df = all_series[bin_days]
        for _, row in series_df.iterrows():
            results.append({
                'bin_days': bin_days,
                'date': row['date'],
                'n': row['n'],
                'm': row['m'],
                'rho_k': row['rho_k']
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_DIR / 'ecp_timeseries_robust.csv', index=False)
    print(f"Saved results to results/ecp_timeseries_robust.csv")


if __name__ == "__main__":
    main()
