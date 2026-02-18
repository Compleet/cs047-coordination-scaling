#!/usr/bin/env python3
"""
Cheeger Constant Validation (SI Appendix E.1)

Paper: "Coordination Scaling: Two Universality Classes Under Capacity Constraint"

Tests the Class M bottleneck topology prediction: coordination networks
should exhibit lower Cheeger constant (more pronounced bottlenecks) than
degree-matched random graphs.

Uses the Cheeger inequality lambda_2 / 2 <= phi <= sqrt(2 * lambda_2),
where lambda_2 is the algebraic connectivity (second-smallest eigenvalue
of the normalized Laplacian) and phi is the Cheeger constant. Compares
the real network lambda_2 against configuration model (same degree
sequence) and Erdos-Renyi baselines.

Networks analyzed:
  - Bitcoin OTC (soc-sign-bitcoinotc)
  - Bitcoin Alpha (soc-sign-bitcoinalpha)

Outputs:
  - figures/cheeger_comparison.png
  - results/cheeger_results.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
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


def compute_algebraic_connectivity(G):
    """
    Compute algebraic connectivity lambda_2 via normalized Laplacian.

    Converts to undirected graph, extracts the largest connected component,
    and computes the second-smallest eigenvalue of the normalized Laplacian
    using the Lanczos algorithm.

    Cheeger inequality: lambda_2 / 2 <= phi <= sqrt(2 * lambda_2)
    """
    G_u = G.to_undirected()

    if not nx.is_connected(G_u):
        largest_cc = max(nx.connected_components(G_u), key=len)
        G_u = G_u.subgraph(largest_cc).copy()

    n = G_u.number_of_nodes()
    if n < 3:
        return np.nan, None

    try:
        lambda2 = nx.algebraic_connectivity(G_u, normalized=True, method='lanczos')
        return lambda2, G_u
    except Exception:
        return np.nan, G_u


def generate_config_model(G, n_samples=10):
    """Generate configuration model graphs with the same degree sequence."""
    degree_sequence = [d for n, d in G.degree()]
    lambdas = []

    for i in range(n_samples):
        try:
            G_rand = nx.configuration_model(degree_sequence)
            G_rand = nx.Graph(G_rand)  # Remove multi-edges
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

            if not nx.is_connected(G_rand):
                largest_cc = max(nx.connected_components(G_rand), key=len)
                G_rand = G_rand.subgraph(largest_cc).copy()

            if G_rand.number_of_nodes() > 2:
                l2 = nx.algebraic_connectivity(G_rand, normalized=True, method='lanczos')
                lambdas.append(l2)
        except Exception:
            continue

    return lambdas


def generate_er_random(n, m, n_samples=10):
    """Generate Erdos-Renyi random graphs with matching size and density."""
    lambdas = []
    p = 2 * m / (n * (n - 1))

    for _ in range(n_samples):
        try:
            G_rand = nx.erdos_renyi_graph(n, p)
            if not nx.is_connected(G_rand):
                largest_cc = max(nx.connected_components(G_rand), key=len)
                G_rand = G_rand.subgraph(largest_cc).copy()

            if G_rand.number_of_nodes() > 2:
                l2 = nx.algebraic_connectivity(G_rand, normalized=True, method='lanczos')
                lambdas.append(l2)
        except Exception:
            continue

    return lambdas


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cheeger Constant Validation (SI Appendix E.1)")
    print("=" * 70)
    print()
    print("Testing: Class M networks have lower Cheeger constant than random")
    print("Cheeger inequality: lambda_2 / 2 <= phi <= sqrt(2 * lambda_2)")
    print()

    networks = [
        ("Bitcoin_OTC", load_bitcoin("soc-sign-bitcoinotc.csv")),
        ("Bitcoin_Alpha", load_bitcoin("soc-sign-bitcoinalpha.csv")),
    ]

    results = []

    for name, G in networks:
        print(f"\nAnalyzing {name}...")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Compute for real network
        lambda2_real, G_u = compute_algebraic_connectivity(G)
        if np.isnan(lambda2_real):
            print(f"  Failed to compute lambda_2")
            continue

        n = G_u.number_of_nodes()
        m = G_u.number_of_edges()

        phi_lower = lambda2_real / 2
        phi_upper = np.sqrt(2 * lambda2_real)

        print(f"  lambda_2 (real): {lambda2_real:.6f}")
        print(f"  Cheeger bounds: [{phi_lower:.6f}, {phi_upper:.6f}]")

        # Compare to configuration model (same degree sequence)
        print(f"  Generating 50 configuration model graphs...")
        lambda2_config = generate_config_model(G_u, n_samples=50)

        # Compare to ER random graphs
        print(f"  Generating 50 ER random graphs...")
        lambda2_er = generate_er_random(n, m, n_samples=50)

        if lambda2_config:
            config_mean = np.mean(lambda2_config)
            config_std = np.std(lambda2_config)
            print(f"  lambda_2 (config model): {config_mean:.6f} +/- {config_std:.6f}")
            ratio_config = lambda2_real / config_mean if config_mean > 0 else np.nan
        else:
            config_mean, config_std, ratio_config = np.nan, np.nan, np.nan

        if lambda2_er:
            er_mean = np.mean(lambda2_er)
            er_std = np.std(lambda2_er)
            print(f"  lambda_2 (ER random): {er_mean:.6f} +/- {er_std:.6f}")
            ratio_er = lambda2_real / er_mean if er_mean > 0 else np.nan
        else:
            er_mean, er_std, ratio_er = np.nan, np.nan, np.nan

        # Lower lambda_2 = lower Cheeger = more bottlenecks = Class M signature
        is_class_m = lambda2_real < config_mean if not np.isnan(config_mean) else None

        results.append({
            'network': name,
            'n_nodes': n,
            'n_edges': m,
            'lambda2_real': lambda2_real,
            'phi_lower': phi_lower,
            'phi_upper': phi_upper,
            'lambda2_config_mean': config_mean,
            'lambda2_config_std': config_std,
            'lambda2_er_mean': er_mean,
            'lambda2_er_std': er_std,
            'ratio_vs_config': ratio_config,
            'ratio_vs_er': ratio_er,
            'class_m_signature': is_class_m
        })

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    class_m_count = 0
    for r in results:
        print(f"\n{r['network']}:")
        print(f"  lambda_2 real:   {r['lambda2_real']:.6f}")
        print(f"  lambda_2 config: {r['lambda2_config_mean']:.6f} +/- {r['lambda2_config_std']:.6f}")
        print(f"  lambda_2 ER:     {r['lambda2_er_mean']:.6f} +/- {r['lambda2_er_std']:.6f}")
        print(f"  Cheeger phi: [{r['phi_lower']:.4f}, {r['phi_upper']:.4f}]")

        if r['class_m_signature']:
            print(f"  Lower connectivity than random: Class M signature detected")
            class_m_count += 1
        elif r['class_m_signature'] is False:
            print(f"  Higher connectivity than random: Class M signature not detected")
        else:
            print(f"  Inconclusive")

    print()
    print("=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if class_m_count == len(results):
        print()
        print("Cheeger hypothesis supported: All tested networks show")
        print("  lower algebraic connectivity (more bottlenecks) than")
        print("  degree-matched random networks.")
        print()
        print("  Consistent with Class M bottleneck topology (SI Appendix E.1).")
    elif class_m_count > 0:
        print(f"\nPartial support: {class_m_count}/{len(results)} networks")
        print("  show Class M Cheeger signature.")
    else:
        print("\nNot supported: Networks do not show expected bottleneck topology.")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'cheeger_results.csv', index=False)
    print(f"\nSaved results to {RESULTS_DIR / 'cheeger_results.csv'}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results))
    width = 0.25

    real_vals = [r['lambda2_real'] for r in results]
    config_vals = [r['lambda2_config_mean'] for r in results]
    config_errs = [r['lambda2_config_std'] for r in results]
    er_vals = [r['lambda2_er_mean'] for r in results]
    er_errs = [r['lambda2_er_std'] for r in results]

    ax.bar(x - width, real_vals, width, label='Real network', color='red', alpha=0.8)
    ax.bar(x, config_vals, width, yerr=config_errs, label='Config model', color='blue', alpha=0.6)
    ax.bar(x + width, er_vals, width, yerr=er_errs, label='ER random', color='green', alpha=0.6)

    ax.set_ylabel('Algebraic Connectivity lambda_2')
    ax.set_xlabel('Network')
    ax.set_xticks(x)
    ax.set_xticklabels([r['network'] for r in results])
    ax.legend()
    ax.set_title('Cheeger Constant: Real vs. Null Model Algebraic Connectivity')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cheeger_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {FIG_DIR / 'cheeger_comparison.png'}")


if __name__ == "__main__":
    main()
