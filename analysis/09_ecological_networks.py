#!/usr/bin/env python3
"""
Ecological Network Analysis (Cross-Domain Validation)
Paper: "Two Universality Classes of Coordination Scaling Under Capacity Constraint"

Tests the Class M/T classification framework on ecological interaction networks.
Hypothesis: mutualistic networks (pollination, seed dispersal) exhibit Class T
signatures, while competitive networks (food webs) exhibit Class M signatures.

Classification uses three spectral/structural indicators:
  - Spectral concentration rho_k (SVD energy in top 5% modes)
  - Degree exponent gamma (MLE power-law tail)
  - Gini coefficient of degree sequence

Also computes nestedness (NODF) and algebraic connectivity (lambda_2).

Data: Web of Life database (www.web-of-life.es), JSON format.

Outputs:
  - figures/ecological_networks.png
  - results/ecological_networks.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
import json
from scipy.linalg import svd, eigh
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ecological_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_web_of_life_network(filepath):
    """Load network from Web of Life JSON format."""
    with open(filepath) as f:
        data = json.load(f)

    G = nx.Graph()

    for node in data.get('nodes', []):
        G.add_node(node['nodeid'],
                   name=node.get('name', ''),
                   group=node.get('group', ''))

    for link in data.get('links', []):
        source = str(link.get('source', link.get('sourceid', '')))
        target = str(link.get('target', link.get('targetid', '')))
        weight = link.get('value', link.get('weight', 1))
        if source and target:
            G.add_edge(source, target, weight=weight)

    return G


def compute_spectral_concentration(G, k_frac=0.05):
    """Compute spectral concentration rho_k (fraction of energy in top k modes)."""
    if G.number_of_nodes() < 10:
        return np.nan

    A = nx.adjacency_matrix(G).astype(float).toarray()
    try:
        U, s, Vt = svd(A, full_matrices=False)
    except Exception:
        return np.nan

    n = len(s)
    k = max(1, int(n * k_frac))

    total_energy = np.sum(s**2)
    top_k_energy = np.sum(s[:k]**2)

    return top_k_energy / total_energy if total_energy > 0 else np.nan


def compute_degree_exponent(G):
    """Estimate power-law exponent gamma via MLE (Hill estimator)."""
    degrees = [d for n, d in G.degree() if d > 0]
    if len(degrees) < 10:
        return np.nan

    k_min = max(1, np.percentile(degrees, 10))
    filtered = [k for k in degrees if k >= k_min]

    if len(filtered) < 5:
        return np.nan

    gamma = 1 + len(filtered) / np.sum(np.log(np.array(filtered) / k_min))
    return gamma


def compute_gini(values):
    """Compute Gini coefficient of a distribution."""
    values = np.array(values)
    values = values[values > 0]
    if len(values) < 2:
        return np.nan

    sorted_values = np.sort(values)
    n = len(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini


def compute_algebraic_connectivity(G):
    """Compute algebraic connectivity lambda_2 (second-smallest Laplacian eigenvalue)."""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    if G.number_of_nodes() < 3:
        return np.nan

    L = nx.laplacian_matrix(G).astype(float).toarray()
    try:
        eigenvalues = np.sort(np.real(eigh(L, eigvals_only=True)))
        return eigenvalues[1] if len(eigenvalues) > 1 else np.nan
    except Exception:
        return np.nan


def compute_nestedness(G):
    """
    Compute nestedness (NODF) for the adjacency matrix.

    Higher nestedness indicates a hierarchical interaction pattern characteristic
    of mutualistic networks (Almeida-Neto et al. 2008).
    """
    if G.number_of_nodes() < 5:
        return np.nan

    A = nx.adjacency_matrix(G).astype(float).toarray()

    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)

    row_order = np.argsort(row_sums)[::-1]
    col_order = np.argsort(col_sums)[::-1]

    A_sorted = A[row_order][:, col_order]

    n_rows, n_cols = A_sorted.shape

    nodf_rows = 0
    n_pairs_rows = 0
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if row_sums[row_order[i]] > row_sums[row_order[j]] and row_sums[row_order[j]] > 0:
                overlap = np.sum((A_sorted[i] > 0) & (A_sorted[j] > 0))
                nodf_rows += overlap / row_sums[row_order[j]]
                n_pairs_rows += 1

    nodf_cols = 0
    n_pairs_cols = 0
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if col_sums[col_order[i]] > col_sums[col_order[j]] and col_sums[col_order[j]] > 0:
                overlap = np.sum((A_sorted[:, i] > 0) & (A_sorted[:, j] > 0))
                nodf_cols += overlap / col_sums[col_order[j]]
                n_pairs_cols += 1

    total_pairs = n_pairs_rows + n_pairs_cols
    if total_pairs == 0:
        return np.nan

    nodf = 100 * (nodf_rows + nodf_cols) / total_pairs
    return nodf


def analyze_network(G, name, network_type):
    """Compute all spectral and structural metrics for a single network."""
    print(f"\nAnalyzing {name} ({network_type})...")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < 10 or m < 5:
        print(f"  Too small: {n} nodes, {m} edges - skipping")
        return None

    print(f"  Size: {n} nodes, {m} edges")

    # Use largest connected component
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(f"  LCC: {G.number_of_nodes()} nodes")

    # Compute metrics
    rho_k = compute_spectral_concentration(G)
    gamma = compute_degree_exponent(G)
    lambda2 = compute_algebraic_connectivity(G)
    gini = compute_gini([d for n, d in G.degree()])
    nestedness = compute_nestedness(G)

    # Class determination by majority vote on three indicators
    class_m_votes = 0
    if gamma is not None and gamma < 2.5:
        class_m_votes += 1
    if gini is not None and gini > 0.5:
        class_m_votes += 1
    if rho_k is not None and rho_k > 0.3:
        class_m_votes += 1

    classification = "Class M" if class_m_votes >= 2 else "Class T"

    rho_str = f"{rho_k:.3f}" if rho_k is not None and not np.isnan(rho_k) else "N/A"
    gamma_str = f"{gamma:.2f}" if gamma is not None and not np.isnan(gamma) else "N/A"
    gini_str = f"{gini:.3f}" if gini is not None and not np.isnan(gini) else "N/A"
    nest_str = f"{nestedness:.1f}" if nestedness is not None and not np.isnan(nestedness) else "N/A"
    print(f"  rho_k={rho_str}, gamma={gamma_str}, Gini={gini_str}, Nest={nest_str}")
    print(f"  -> {classification} ({class_m_votes}/3 Class M votes)")

    return {
        'network': name,
        'type': network_type,
        'n': n,
        'm': m,
        'rho_k': rho_k,
        'gamma': gamma,
        'lambda2': lambda2,
        'gini': gini,
        'nestedness': nestedness,
        'class_m_votes': class_m_votes,
        'classification': classification
    }


def main():
    print("=" * 70)
    print("Ecological Network Analysis (Cross-Domain Validation)")
    print("=" * 70)
    print()
    print("Hypothesis: Mutualistic -> Class T, Competitive -> Class M")
    print()

    # Define networks to analyze
    networks = [
        ('eco_mutualistic_1.json', 'Pollination 1', 'Mutualistic'),
        ('eco_mutualistic_2.json', 'Pollination 2', 'Mutualistic'),
        ('eco_mutualistic_3.json', 'Pollination 3', 'Mutualistic'),
        ('eco_seed_dispersal_1.json', 'Seed Dispersal 1', 'Mutualistic'),
        ('eco_seed_dispersal_2.json', 'Seed Dispersal 2', 'Mutualistic'),
        ('eco_foodweb_1.json', 'Food Web 1', 'Competitive'),
        ('eco_foodweb_2.json', 'Food Web 2', 'Competitive'),
    ]

    all_results = []

    for filename, name, net_type in networks:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"\n{filename} not found - skipping")
            continue

        try:
            G = load_web_of_life_network(filepath)
            result = analyze_network(G, name, net_type)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    if not all_results:
        print("\nNo networks analyzed successfully.")
        print(f"Expected data files in: {DATA_DIR}")
        return

    df = pd.DataFrame(all_results)

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print(f"{'Network':18s} {'Type':12s} {'rho_k':>6s} {'gamma':>6s} {'Gini':>6s}  {'Class'}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['network']:18s} {row['type']:12s} {row['rho_k']:.3f}  {row['gamma']:.2f}  {row['gini']:.3f}  {row['classification']}")

    print()

    # Group statistics
    mutualistic = df[df['type'] == 'Mutualistic']
    competitive = df[df['type'] == 'Competitive']

    print("BY INTERACTION TYPE:")
    if len(mutualistic) > 0:
        print(f"  Mutualistic: mean rho_k = {mutualistic['rho_k'].mean():.3f}, "
              f"mean gamma = {mutualistic['gamma'].mean():.2f}, "
              f"Class T: {(mutualistic['classification'] == 'Class T').sum()}/{len(mutualistic)}")
    if len(competitive) > 0:
        print(f"  Competitive: mean rho_k = {competitive['rho_k'].mean():.3f}, "
              f"mean gamma = {competitive['gamma'].mean():.2f}, "
              f"Class M: {(competitive['classification'] == 'Class M').sum()}/{len(competitive)}")

    print()

    # Hypothesis test
    mutualistic_class_t = (mutualistic['classification'] == 'Class T').sum() if len(mutualistic) > 0 else 0
    competitive_class_m = (competitive['classification'] == 'Class M').sum() if len(competitive) > 0 else 0

    total_mutualistic = len(mutualistic)
    total_competitive = len(competitive)

    print("HYPOTHESIS TEST:")
    print(f"  Mutualistic -> Class T: {mutualistic_class_t}/{total_mutualistic}")
    print(f"  Competitive -> Class M: {competitive_class_m}/{total_competitive}")

    if total_mutualistic > 0 and total_competitive > 0:
        if mutualistic_class_t > total_mutualistic / 2 and competitive_class_m > total_competitive / 2:
            print("  HYPOTHESIS SUPPORTED")
        else:
            print("  MIXED RESULTS")
    else:
        print("  INSUFFICIENT DATA")

    # Visualization
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: rho_k by type
    ax1 = axes[0]
    if len(mutualistic) > 0:
        ax1.bar(0, mutualistic['rho_k'].mean(), yerr=mutualistic['rho_k'].std(),
                color='green', alpha=0.7, label='Mutualistic', capsize=5)
    if len(competitive) > 0:
        ax1.bar(1, competitive['rho_k'].mean(), yerr=competitive['rho_k'].std(),
                color='red', alpha=0.7, label='Competitive', capsize=5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Mutualistic', 'Competitive'])
    ax1.set_ylabel('Spectral Concentration $\\rho_k$')
    ax1.set_title('Spectral Concentration by Interaction Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: gamma by type
    ax2 = axes[1]
    if len(mutualistic) > 0:
        ax2.bar(0, mutualistic['gamma'].mean(), yerr=mutualistic['gamma'].std(),
                color='green', alpha=0.7, capsize=5)
    if len(competitive) > 0:
        ax2.bar(1, competitive['gamma'].mean(), yerr=competitive['gamma'].std(),
                color='red', alpha=0.7, capsize=5)
    ax2.axhline(2, color='gray', linestyle='--', alpha=0.5, label='$\\gamma=2$ threshold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Mutualistic', 'Competitive'])
    ax2.set_ylabel('Degree Exponent $\\gamma$')
    ax2.set_title('Degree Distribution by Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Scatter (rho_k vs Gini, colored by type)
    ax3 = axes[2]
    colors = {'Mutualistic': 'green', 'Competitive': 'red'}
    for _, row in df.iterrows():
        ax3.scatter(row['rho_k'], row['gini'], s=100,
                    c=colors.get(row['type'], 'gray'),
                    label=row['type'] if row['type'] not in ax3.get_legend_handles_labels()[1] else '',
                    alpha=0.7, edgecolors='black')
        ax3.annotate(row['network'][:10], (row['rho_k'], row['gini']),
                     fontsize=8, alpha=0.7)

    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0.3, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Spectral Concentration $\\rho_k$')
    ax3.set_ylabel('Gini Coefficient')
    ax3.set_title('Phase Space: Spectral Concentration vs Inequality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ecological_networks.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {FIG_DIR / 'ecological_networks.png'}")

    df.to_csv(RESULTS_DIR / 'ecological_networks.csv', index=False)
    print(f"Saved results: {RESULTS_DIR / 'ecological_networks.csv'}")

    # Conclusion
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Ecological networks provide a natural test of the Class M/T framework:")
    print("- Mutualistic networks (pollination, seed dispersal) show cooperation")
    print("- Food webs show competitive/predatory dynamics")


if __name__ == "__main__":
    main()
