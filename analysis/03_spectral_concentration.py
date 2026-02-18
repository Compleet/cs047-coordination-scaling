#!/usr/bin/env python3
"""
Spectral Concentration (ECP) Direct Test (Section 2.3)

Paper: "Coordination Scaling: Two Universality Classes Under Capacity Constraint"

Tests the Effective Concentration Property (ECP) hypothesis: that coordination
networks exhibit rho_k -> 1 for k << N, i.e., the adjacency matrix singular
value spectrum concentrates much faster than expected under null models.

Computes the cumulative spectral concentration rho_k = sum(sigma_i^2, i=1..k)
/ ||A||_F^2 for the real network and compares against two null baselines:
  1. Maslov-Sneppen degree-preserving rewiring (20 realizations)
  2. Erdos-Renyi G(n, m) random graphs (20 realizations)

Networks analyzed:
  - Bitcoin OTC (soc-sign-bitcoinotc)
  - Bitcoin Alpha (soc-sign-bitcoinalpha)

Outputs:
  - figures/spectral_concentration.png
  - results/spectral_summary.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_bitcoin(filename):
    """Load Bitcoin trust network from CSV."""
    df = pd.read_csv(DATA_DIR / filename, header=None,
                     names=['source', 'target', 'rating', 'time'])
    G = nx.DiGraph()
    G.add_edges_from(zip(df['source'], df['target']))
    return G


def graph_to_sparse_adjacency(G):
    """Convert NetworkX graph to sparse CSR adjacency matrix."""
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    n = len(nodes)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A


def compute_spectral_concentration(A, k_max=50):
    """
    Compute rho_k = sum(sigma_i^2, i=1..k) / ||A||_F^2.

    Uses truncated SVD to obtain the top-k singular values and normalizes
    by the squared Frobenius norm of A.
    """
    n = A.shape[0]
    k_compute = min(k_max, n - 2)

    try:
        U, s, Vt = svds(A.astype(float), k=k_compute)
        s = np.sort(s)[::-1]
    except Exception as e:
        print(f"  SVD failed: {e}")
        return None, None

    total_variance = sparse.linalg.norm(A.astype(float), 'fro')**2
    cumsum_variance = np.cumsum(s**2)
    rho_k = cumsum_variance / total_variance
    k_values = np.arange(1, len(s) + 1)

    return k_values, rho_k


def maslov_sneppen_rewire(G, n_swaps=None):
    """Degree-preserving random rewiring (Maslov & Sneppen, 2002)."""
    G_rewired = G.copy()
    edges = list(G_rewired.edges())
    n_edges = len(edges)

    if n_swaps is None:
        n_swaps = n_edges * 5

    successful = 0
    attempts = 0
    max_attempts = n_swaps * 10

    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = np.random.choice(n_edges, 2, replace=False)
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]

        if len({u1, v1, u2, v2}) < 4:
            continue
        if G_rewired.has_edge(u1, v2) or G_rewired.has_edge(u2, v1):
            continue
        if u1 == v2 or u2 == v1:
            continue

        G_rewired.remove_edge(u1, v1)
        G_rewired.remove_edge(u2, v2)
        G_rewired.add_edge(u1, v2)
        G_rewired.add_edge(u2, v1)
        edges[idx1] = (u1, v2)
        edges[idx2] = (u2, v1)
        successful += 1

    return G_rewired


def generate_er_graph(n, m):
    """Generate a directed Erdos-Renyi G(n, m) random graph."""
    return nx.gnm_random_graph(n, m, directed=True)


def analyze_network(G, name, n_null=20, k_max=30):
    """
    Compute spectral concentration for a network and null models.

    Returns a dict with rho_k curves for the real network, mean/CI for
    Maslov-Sneppen rewired nulls, mean for ER random nulls, and p-values.
    """
    print(f"\nAnalyzing {name}...")
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n:,}, Edges: {m:,}")

    A_real = graph_to_sparse_adjacency(G)
    print("  Computing SVD for real network...")
    k_real, rho_real = compute_spectral_concentration(A_real, k_max=k_max)

    if rho_real is None:
        return None

    print(f"  Generating {n_null} null models (degree-preserving rewiring)...")
    rho_null_all = []

    for i in range(n_null):
        if (i + 1) % 5 == 0:
            print(f"    Null model {i+1}/{n_null}...")
        G_null = maslov_sneppen_rewire(G, n_swaps=m)
        A_null = graph_to_sparse_adjacency(G_null)
        k_null, rho_null = compute_spectral_concentration(A_null, k_max=k_max)
        if rho_null is not None:
            rho_null_all.append(rho_null)

    rho_null_all = np.array(rho_null_all)

    print(f"  Generating {n_null} ER random graphs...")
    rho_er_all = []

    for i in range(n_null):
        G_er = generate_er_graph(n, m)
        A_er = graph_to_sparse_adjacency(G_er)
        k_er, rho_er = compute_spectral_concentration(A_er, k_max=k_max)
        if rho_er is not None:
            rho_er_all.append(rho_er)

    rho_er_all = np.array(rho_er_all)

    # Statistics
    rho_null_mean = np.mean(rho_null_all, axis=0)
    rho_null_5th = np.percentile(rho_null_all, 5, axis=0)
    rho_null_95th = np.percentile(rho_null_all, 95, axis=0)
    rho_er_mean = np.mean(rho_er_all, axis=0)

    # P-values (fraction of null realizations with rho >= real)
    p_values = np.array([np.mean(rho_null_all[:, j] >= rho_real[j]) for j in range(len(k_real))])

    return {
        'name': name, 'n': n, 'm': m, 'k': k_real,
        'rho_real': rho_real,
        'rho_null_mean': rho_null_mean,
        'rho_null_5th': rho_null_5th,
        'rho_null_95th': rho_null_95th,
        'rho_er_mean': rho_er_mean,
        'p_values': p_values
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Spectral Concentration (ECP) Direct Test (Section 2.3)")
    print("=" * 70)
    print("\nTesting: rho_k -> 1 for k << N in coordination networks")
    print("Compares real networks against degree-preserving and ER null models.\n")

    print("Loading networks...")

    networks = {}
    print("  Bitcoin OTC...")
    networks['Bitcoin_OTC'] = load_bitcoin("soc-sign-bitcoinotc.csv")

    print("  Bitcoin Alpha...")
    networks['Bitcoin_Alpha'] = load_bitcoin("soc-sign-bitcoinalpha.csv")

    results_all = []

    for name, G in networks.items():
        results = analyze_network(G, name, n_null=20, k_max=25)
        if results is not None:
            results_all.append(results)

    # Plot
    print("\nCreating plot...")

    fig, axes = plt.subplots(1, len(results_all), figsize=(6 * len(results_all), 5))
    if len(results_all) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_all):
        k_norm = res['k'] / res['n']

        ax.plot(k_norm, res['rho_real'], 'b-', linewidth=2, label='Real network')
        ax.fill_between(k_norm, res['rho_null_5th'], res['rho_null_95th'],
                        color='orange', alpha=0.3, label='Null 90% CI')
        ax.plot(k_norm, res['rho_null_mean'], 'orange', linestyle='--', label='Null mean')
        ax.plot(k_norm, res['rho_er_mean'], 'g:', label='ER random')

        ax.set_xlabel('k / N')
        ax.set_ylabel('rho_k (cumulative variance)')
        ax.set_title(f"{res['name']} (n={res['n']:,})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(k_norm)])
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'spectral_concentration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {FIG_DIR / 'spectral_concentration.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for res in results_all:
        k_test = max(1, int(0.01 * res['n']))
        k_idx = min(k_test - 1, len(res['k']) - 1)

        rho_real = res['rho_real'][k_idx]
        rho_null = res['rho_null_mean'][k_idx]
        p_val = res['p_values'][k_idx]
        significant = p_val < 0.01

        print(f"\n{res['name']}:")
        print(f"  At k={k_test} (k/N ~ 0.01):")
        print(f"    Real rho_k:  {rho_real:.4f}")
        print(f"    Null rho_k:  {rho_null:.4f}")
        print(f"    Delta = {rho_real - rho_null:.4f}")
        print(f"    p-value:   {p_val:.4f}")
        print(f"    Significant (p<0.01): {'YES' if significant else 'NO'}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    higher_concentration = []
    for res in results_all:
        k_idx = min(max(1, int(0.01 * res['n'])) - 1, len(res['k']) - 1)
        if res['rho_real'][k_idx] > res['rho_null_mean'][k_idx]:
            higher_concentration.append(res['name'])

    if len(higher_concentration) == len(results_all):
        print("\nECP supported: All networks show higher spectral")
        print("  concentration than null models at k << N.")
    elif higher_concentration:
        print(f"\nPartial support: {len(higher_concentration)}/{len(results_all)} networks")
        print(f"  show higher concentration: {higher_concentration}")
    else:
        print("\nECP not supported: Networks do not show higher")
        print("  concentration than null models.")

    # Save CSV
    summary_rows = []
    for res in results_all:
        k_idx = min(max(1, int(0.01 * res['n'])) - 1, len(res['k']) - 1)
        summary_rows.append({
            'Network': res['name'],
            'N': res['n'],
            'M': res['m'],
            'rho_real_1pct': res['rho_real'][k_idx],
            'rho_null_1pct': res['rho_null_mean'][k_idx],
            'p_value': res['p_values'][k_idx],
            'higher_than_null': res['rho_real'][k_idx] > res['rho_null_mean'][k_idx]
        })

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / 'spectral_summary.csv', index=False)
    print(f"\nSaved summary to {RESULTS_DIR / 'spectral_summary.csv'}")

    return results_all


if __name__ == '__main__':
    results = main()
