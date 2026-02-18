#!/usr/bin/env python3
"""
ECP k-Threshold Sensitivity Analysis
=====================================
Paper: "Coordination Costs and Scaling Laws in Large-Scale Software Teams"

Tests the robustness of the Eigenmode Concentration Property (ECP) to the
choice of k-threshold. The ECP (Section 5, Definition 3) measures spectral
concentration rho_k = sum_{i=1}^k sigma_i^2 / sum_{j=1}^N sigma_j^2. The
default k = 1% of N is validated here by sweeping k = 0.5%, 1%, 2%, 5%, 10%
and comparing against degree-preserving null models.

Data:
  - Bitcoin OTC trust network (soc-sign-bitcoinotc.csv)
  - Bitcoin Alpha trust network (soc-sign-bitcoinalpha.csv)

Output:
  - figures/ecp_sensitivity.png
  - results/ecp_sensitivity.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import svd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIGURE_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_bitcoin(filename):
    """Load Bitcoin trust network."""
    df = pd.read_csv(DATA_DIR / filename, header=None,
                     names=['source', 'target', 'rating', 'time'])
    G = nx.DiGraph()
    G.add_edges_from(zip(df['source'], df['target']))
    return G


def compute_spectral_concentration(G, k_frac):
    """
    Compute spectral concentration rho_k at given k fraction.
    rho_k = sum_{i=1}^k sigma_i^2 / sum_{j=1}^N sigma_j^2
    """
    n = G.number_of_nodes()
    k = max(1, int(n * k_frac))

    # Get adjacency matrix and compute SVD
    A = nx.adjacency_matrix(G).astype(float).toarray()
    U, s, Vt = svd(A, full_matrices=False)

    total_energy = np.sum(s**2)
    top_k_energy = np.sum(s[:k]**2)

    rho_k = top_k_energy / total_energy if total_energy > 0 else 0
    return rho_k, k


def generate_null_model(G, method='rewire'):
    """Generate null model via edge rewiring (degree-preserving)."""
    G_null = G.copy()
    edges = list(G_null.edges())
    n_rewires = len(edges) * 2

    for _ in range(n_rewires):
        if len(edges) < 2:
            break
        # Pick two random edges
        i1, i2 = np.random.choice(len(edges), 2, replace=False)
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]

        # Rewire: (u1,v1), (u2,v2) -> (u1,v2), (u2,v1)
        if u1 != v2 and u2 != v1 and not G_null.has_edge(u1, v2) and not G_null.has_edge(u2, v1):
            G_null.remove_edge(u1, v1)
            G_null.remove_edge(u2, v2)
            G_null.add_edge(u1, v2)
            G_null.add_edge(u2, v1)
            edges[i1] = (u1, v2)
            edges[i2] = (u2, v1)

    return G_null


def main():
    print("=" * 70)
    print("ECP k-Threshold Sensitivity Analysis (Section 5)")
    print("=" * 70)
    print()
    print("Question: Is the ECP signal robust to the choice of k?")
    print("Testing k = 0.5%, 1%, 2%, 5%, 10% of N")
    print()

    networks = [
        ("Bitcoin_OTC", "soc-sign-bitcoinotc.csv"),
        ("Bitcoin_Alpha", "soc-sign-bitcoinalpha.csv"),
    ]

    k_fractions = [0.005, 0.01, 0.02, 0.05, 0.10]
    n_null = 10  # Number of null models

    all_results = []

    max_nodes = 1500  # Sample for speed

    for name, filename in networks:
        print(f"Analyzing {name}...")

        G = load_bitcoin(filename)
        n_orig = G.number_of_nodes()

        # Sample if too large
        if n_orig > max_nodes:
            nodes = list(G.nodes())
            np.random.seed(42)
            sampled = np.random.choice(nodes, max_nodes, replace=False)
            G = G.subgraph(sampled).copy()
            print(f"  Original N = {n_orig}, sampled to {max_nodes}")

        n = G.number_of_nodes()
        print(f"  Working with N = {n} nodes")
        print()

        for k_frac in k_fractions:
            k = max(1, int(n * k_frac))

            # Real network
            rho_real, _ = compute_spectral_concentration(G, k_frac)

            # Null models
            rho_nulls = []
            for i in range(n_null):
                try:
                    G_null = generate_null_model(G)
                    rho_null, _ = compute_spectral_concentration(G_null, k_frac)
                    rho_nulls.append(rho_null)
                except:
                    pass

            rho_null_mean = np.mean(rho_nulls) if rho_nulls else np.nan
            rho_null_std = np.std(rho_nulls) if rho_nulls else np.nan

            # Effect size
            if rho_null_std > 0:
                z_score = (rho_real - rho_null_mean) / rho_null_std
                p_value = 1 - len([r for r in rho_nulls if r < rho_real]) / len(rho_nulls)
            else:
                z_score = np.nan
                p_value = np.nan

            print(f"  k = {k_frac*100:.1f}% (k={k}): rho_real = {rho_real:.4f}, "
                  f"rho_null = {rho_null_mean:.4f} +/- {rho_null_std:.4f}, "
                  f"z = {z_score:.2f}")

            all_results.append({
                'network': name,
                'n': n,
                'k_frac': k_frac,
                'k': k,
                'rho_real': rho_real,
                'rho_null_mean': rho_null_mean,
                'rho_null_std': rho_null_std,
                'z_score': z_score,
                'p_value': p_value,
                'excess': rho_real - rho_null_mean
            })

        print()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Summary
    print("=" * 70)
    print("SUMMARY: ECP Signal Across k Values")
    print("=" * 70)
    print()

    for name in df['network'].unique():
        df_net = df[df['network'] == name]
        print(f"{name}:")
        print(f"  k%     k    rho_real   rho_null   Excess   z-score")
        print(f"  " + "-" * 50)
        for _, row in df_net.iterrows():
            print(f"  {row['k_frac']*100:4.1f}%  {row['k']:4d}  "
                  f"{row['rho_real']:.4f}   {row['rho_null_mean']:.4f}   "
                  f"{row['excess']:+.4f}  {row['z_score']:+6.2f}")
        print()

    # Key finding
    all_z_positive = all(df['z_score'] > 0)
    all_significant = all(df['z_score'] > 2)

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if all_z_positive:
        print("[+] ECP signal (rho_real > rho_null) is consistent across all k values")
    else:
        print("[!] ECP signal varies with k")

    if all_significant:
        print("[+] All k values show significant excess concentration (z > 2)")
    else:
        n_sig = sum(df['z_score'] > 2)
        print(f"[!] {n_sig}/{len(df)} tests show significant excess (z > 2)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: rho_k vs k for both networks
    ax1 = axes[0]
    for name in df['network'].unique():
        df_net = df[df['network'] == name]
        k_pcts = df_net['k_frac'] * 100
        ax1.plot(k_pcts, df_net['rho_real'], 'o-', markersize=8, label=f'{name} (real)')
        ax1.fill_between(k_pcts,
                         df_net['rho_null_mean'] - df_net['rho_null_std'],
                         df_net['rho_null_mean'] + df_net['rho_null_std'],
                         alpha=0.3, label=f'{name} (null +/-1 std)')
    ax1.set_xlabel('k (% of N)')
    ax1.set_ylabel('Spectral Concentration rho_k')
    ax1.set_title('ECP: Real vs Null Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Excess concentration
    ax2 = axes[1]
    width = 0.35
    x = np.arange(len(k_fractions))
    for i, name in enumerate(df['network'].unique()):
        df_net = df[df['network'] == name]
        offset = width * (i - 0.5)
        ax2.bar(x + offset, df_net['excess'], width, label=name, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('k (% of N)')
    ax2.set_ylabel('Excess Concentration (rho_real - rho_null)')
    ax2.set_title('ECP Excess vs k')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{k*100:.1f}%' for k in k_fractions])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Z-scores
    ax3 = axes[2]
    for i, name in enumerate(df['network'].unique()):
        df_net = df[df['network'] == name]
        offset = width * (i - 0.5)
        ax3.bar(x + offset, df_net['z_score'], width, label=name, alpha=0.7)
    ax3.axhline(2, color='red', linestyle='--', linewidth=1, label='z=2 (p<0.05)')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('k (% of N)')
    ax3.set_ylabel('Z-score')
    ax3.set_title('Statistical Significance of ECP')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{k*100:.1f}%' for k in k_fractions])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / 'ecp_sensitivity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to figures/ecp_sensitivity.png")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'ecp_sensitivity.csv', index=False)
    print(f"Saved results to results/ecp_sensitivity.csv")


if __name__ == "__main__":
    main()
