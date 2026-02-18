#!/usr/bin/env python3
"""
Citation Network Class M Classification
========================================
Paper: "Two Universality Classes of Coordination Scaling Under Capacity Constraint"

Validates the regime classification framework (Section 8) on an out-of-sample
domain: the HEP-PH citation network from SNAP. Citation networks are expected
to exhibit Class M characteristics (power-law exponent gamma in [1.5, 3.0],
high Gini coefficient, lognormal body with Pareto tail).

Analyses performed:
  1. Power law fitting (in-degree, out-degree) using Clauset-Shalizi-Newman
     methodology (cf. Table 2)
  2. Regime classification using distribution shape, Gini coefficient, and
     degree assortativity (Section 8)
  3. Spectral concentration rho_k vs Erdos-Renyi null model (Section 5)
  4. Cheeger constant approximation via algebraic connectivity (Section 6)

Data:
  - cit-HepPh.txt (SNAP High-Energy Physics citation network)

Output:
  - figures/citation_analysis.png
  - results/citation_results.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
from scipy import sparse
from scipy.sparse.linalg import svds, eigsh
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "citation_networks"
FIGURE_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_hep_ph():
    """Load HEP-PH citation network from SNAP."""
    edges = []
    with open(DATA_DIR / "cit-HepPh.txt") as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def analyze_power_law(G, name):
    """Analyze degree distributions using Clauset-Shalizi-Newman methodology."""
    print(f"\n  Power law analysis for {name}...")

    results = {}

    for deg_type in ['in', 'out']:
        if deg_type == 'in':
            degrees = [d for n, d in G.in_degree() if d > 0]
        else:
            degrees = [d for n, d in G.out_degree() if d > 0]

        degrees = np.array(degrees)

        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        sigma = fit.power_law.sigma

        # Comparisons
        R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
        R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

        results[deg_type] = {
            'gamma': alpha,
            'xmin': xmin,
            'sigma': sigma,
            'n_samples': len(degrees),
            'R_vs_lognormal': R_ln,
            'p_vs_lognormal': p_ln,
            'R_vs_exponential': R_exp,
            'p_vs_exponential': p_exp,
            'fit': fit
        }

        print(f"    {deg_type}-degree: gamma = {alpha:.3f} +/- {sigma:.3f}, xmin = {xmin}")

    return results


def compute_gini(values):
    """Compute Gini coefficient."""
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * np.sum(values))) - (n + 1) / n


def analyze_regime_classifier(G, name):
    """
    Apply the regime classifier from Section 8:
    1. Distribution shape (lognormal body vs pure power law)
    2. Gini coefficient (inequality measure)
    3. Degree assortativity
    """
    print(f"\n  Regime classifier for {name}...")

    # 1. Distribution shape
    in_degrees = np.array([d for n, d in G.in_degree() if d > 0])
    fit = powerlaw.Fit(in_degrees, discrete=True, verbose=False)
    R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

    # Lognormal body + Pareto tail is Class M signature
    if p_ln < 0.05 and R_ln < 0:
        dist_shape = "Lognormal body (Class M)"
    elif p_ln < 0.05 and R_ln > 0:
        dist_shape = "Power law preferred (ambiguous)"
    else:
        dist_shape = "Inconclusive"

    print(f"    Distribution shape: {dist_shape}")

    # 2. Gini coefficient (inequality measure)
    gini = compute_gini(in_degrees)
    print(f"    Gini coefficient: {gini:.3f}")

    # High Gini (> 0.5) suggests multiplicative dynamics (Class M)
    gini_verdict = "Class M" if gini > 0.5 else "Class T"
    print(f"    Gini verdict: {gini_verdict} (threshold 0.5)")

    # 3. Degree correlation (assortativity)
    try:
        assort = nx.degree_assortativity_coefficient(G)
        print(f"    Degree assortativity: {assort:.3f}")
    except:
        assort = np.nan

    return {
        'distribution_shape': dist_shape,
        'gini': gini,
        'gini_verdict': gini_verdict,
        'assortativity': assort,
        'gamma': fit.power_law.alpha
    }


def compute_spectral_concentration(A, k_max=30):
    """Compute spectral concentration rho_k."""
    n = A.shape[0]
    k_compute = min(k_max, n - 2)

    try:
        U, s, Vt = svds(A.astype(float), k=k_compute)
        s = np.sort(s)[::-1]
    except:
        return None, None

    total_variance = sparse.linalg.norm(A.astype(float), 'fro')**2
    rho_k = np.cumsum(s**2) / total_variance

    return np.arange(1, len(s) + 1), rho_k


def graph_to_sparse(G):
    """Convert NetworkX graph to sparse adjacency matrix."""
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    n = len(nodes)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def analyze_spectral(G, name, n_null=10):
    """Spectral analysis comparing real network to Erdos-Renyi null model."""
    print(f"\n  Spectral analysis for {name}...")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Sample if too large
    if n > 10000:
        print(f"    Sampling 10000 nodes from {n}...")
        nodes = list(G.nodes())
        start = np.random.choice(nodes)
        sampled = set([start])
        frontier = [start]

        while len(sampled) < 10000 and frontier:
            current = frontier.pop(0)
            for neighbor in list(G.successors(current)) + list(G.predecessors(current)):
                if neighbor not in sampled:
                    sampled.add(neighbor)
                    frontier.append(neighbor)

        G = G.subgraph(sampled).copy()
        n = G.number_of_nodes()
        m = G.number_of_edges()

    print(f"    Nodes: {n:,}, Edges: {m:,}")

    A = graph_to_sparse(G)
    k_real, rho_real = compute_spectral_concentration(A)

    if rho_real is None:
        return None

    # Generate null models
    print(f"    Generating {n_null} ER random graphs...")
    rho_er = []
    for _ in range(n_null):
        G_er = nx.gnm_random_graph(n, m, directed=True)
        A_er = graph_to_sparse(G_er)
        _, rho = compute_spectral_concentration(A_er)
        if rho is not None:
            rho_er.append(rho)

    rho_er = np.array(rho_er) if rho_er else None
    rho_er_mean = np.mean(rho_er, axis=0) if rho_er is not None else None

    return {
        'n': n, 'm': m,
        'k': k_real, 'rho_real': rho_real,
        'rho_er_mean': rho_er_mean
    }


def compute_cheeger_approximation(G):
    """Compute algebraic connectivity (Cheeger constant approximation)."""
    G_undirected = G.to_undirected()

    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    n = G_undirected.number_of_nodes()
    if n < 100:
        return np.nan

    L = nx.normalized_laplacian_matrix(G_undirected).astype(float)

    try:
        eigenvalues, _ = eigsh(L, k=6, which='SM', tol=1e-6)
        eigenvalues = np.sort(eigenvalues)
        lambda_2 = eigenvalues[1] if eigenvalues[0] < 0.01 else eigenvalues[0]
        return lambda_2
    except:
        return np.nan


def main():
    print("=" * 70)
    print("Citation Networks: Class M Classification (Section 8)")
    print("=" * 70)
    print("\nHypothesis: Citation networks exhibit Class M characteristics")
    print("with gamma in [1.5, 3.0].\n")

    # Load HEP-PH
    print("Loading HEP-PH citation network...")
    G = load_hep_ph()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n:,}, Edges: {m:,}")

    # 1. Power law analysis
    print("\n" + "=" * 50)
    print("1. POWER LAW ANALYSIS (cf. Table 2)")
    print("=" * 50)

    pl_results = analyze_power_law(G, "HEP-PH")

    # 2. Regime classifier
    print("\n" + "=" * 50)
    print("2. REGIME CLASSIFIER (Section 8)")
    print("=" * 50)

    regime_results = analyze_regime_classifier(G, "HEP-PH")

    # 3. Spectral analysis
    print("\n" + "=" * 50)
    print("3. SPECTRAL CONCENTRATION (ECP)")
    print("=" * 50)

    spectral_results = analyze_spectral(G, "HEP-PH", n_null=50)

    # 4. Cheeger constant
    print("\n" + "=" * 50)
    print("4. CHEEGER CONSTANT APPROXIMATION")
    print("=" * 50)

    lambda_2 = compute_cheeger_approximation(G)
    print(f"  Algebraic connectivity lambda_2: {lambda_2:.6f}")
    phi_lower = lambda_2 / 2
    phi_upper = np.sqrt(2 * lambda_2)
    print(f"  Cheeger bounds: [{phi_lower:.6f}, {phi_upper:.6f}]")

    # Create summary plot
    print("\nCreating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: In-degree distribution
    ax = axes[0, 0]
    fit = pl_results['in']['fit']
    fit.plot_ccdf(ax=ax, color='b', linewidth=2, label='Data')
    fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--',
                            label=f'Power Law (gamma={pl_results["in"]["gamma"]:.2f})')
    fit.lognormal.plot_ccdf(ax=ax, color='g', linestyle=':', label='Lognormal')
    ax.set_title(f'HEP-PH In-Degree Distribution\ngamma = {pl_results["in"]["gamma"]:.3f}')
    ax.set_xlabel('Degree k')
    ax.set_ylabel('P(X >= k)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Out-degree distribution
    ax = axes[0, 1]
    fit = pl_results['out']['fit']
    fit.plot_ccdf(ax=ax, color='b', linewidth=2, label='Data')
    fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--',
                            label=f'Power Law (gamma={pl_results["out"]["gamma"]:.2f})')
    ax.set_title(f'HEP-PH Out-Degree Distribution\ngamma = {pl_results["out"]["gamma"]:.3f}')
    ax.set_xlabel('Degree k')
    ax.set_ylabel('P(X >= k)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Spectral concentration
    ax = axes[1, 0]
    if spectral_results:
        k_norm = spectral_results['k'] / spectral_results['n']
        ax.plot(k_norm, spectral_results['rho_real'], 'b-', linewidth=2, label='HEP-PH')
        if spectral_results['rho_er_mean'] is not None:
            ax.plot(k_norm[:len(spectral_results['rho_er_mean'])],
                    spectral_results['rho_er_mean'], 'g--', label='ER random')
        ax.set_xlabel('k / N')
        ax.set_ylabel('rho_k')
        ax.set_title('Spectral Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Spectral analysis failed', ha='center', va='center')

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    CLASSIFICATION SUMMARY
    =====================

    Power Law Exponents:
      In-degree:  gamma = {pl_results['in']['gamma']:.3f} +/- {pl_results['in']['sigma']:.3f}
      Out-degree: gamma = {pl_results['out']['gamma']:.3f} +/- {pl_results['out']['sigma']:.3f}

    Expected range for Class M: [1.5, 3.0]
    In-degree within range: {'YES' if 1.5 <= pl_results['in']['gamma'] <= 3.0 else 'NO'}

    Regime Indicators:
      Gini coefficient: {regime_results['gini']:.3f}
      Distribution: {regime_results['distribution_shape']}

    Cheeger Constant:
      lambda_2 = {lambda_2:.6f}
      phi bounds: [{phi_lower:.6f}, {phi_upper:.6f}]

    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / 'citation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to figures/citation_analysis.png")

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    gamma_in = pl_results['in']['gamma']
    gamma_out = pl_results['out']['gamma']

    print(f"\nHEP-PH Citation Network:")
    print(f"  In-degree gamma:  {gamma_in:.3f} (expected 2.5-3.0 from literature)")
    print(f"  Out-degree gamma: {gamma_out:.3f}")
    print(f"  Gini: {regime_results['gini']:.3f}")

    class_m_votes = 0
    total_votes = 0

    # Vote 1: Gamma in expected range
    total_votes += 1
    if 1.5 <= gamma_in <= 3.0:
        class_m_votes += 1
        print(f"\n  [+] gamma in expected Class M range [1.5, 3.0]")
    else:
        print(f"\n  [-] gamma outside expected range")

    # Vote 2: Lognormal body
    total_votes += 1
    if "Lognormal" in regime_results['distribution_shape']:
        class_m_votes += 1
        print("  [+] Lognormal body (Class M signature)")
    else:
        print(f"  [?] Distribution: {regime_results['distribution_shape']}")

    # Vote 3: High Gini
    total_votes += 1
    if regime_results['gini'] > 0.5:
        class_m_votes += 1
        print("  [+] High Gini (multiplicative dynamics)")
    else:
        print(f"  [-] Low Gini")

    # Vote 4: Spectral concentration
    if spectral_results and spectral_results['rho_er_mean'] is not None:
        total_votes += 1
        k_idx = min(5, len(spectral_results['rho_real']) - 1)
        if spectral_results['rho_real'][k_idx] > spectral_results['rho_er_mean'][k_idx]:
            class_m_votes += 1
            print("  [+] Higher spectral concentration than ER")
        else:
            print("  [-] No spectral concentration advantage")

    print(f"\n  Classification votes: {class_m_votes}/{total_votes} for Class M")

    if class_m_votes >= total_votes / 2:
        print("\n  --> CLASSIFIED AS CLASS M")
    else:
        print("\n  --> CLASSIFICATION UNCERTAIN")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([{
        'Network': 'HEP-PH',
        'gamma_in': gamma_in,
        'gamma_out': gamma_out,
        'gini': regime_results['gini'],
        'lambda_2': lambda_2,
        'class_m_votes': class_m_votes,
        'total_votes': total_votes,
        'classification': 'Class M' if class_m_votes >= total_votes / 2 else 'Uncertain'
    }])
    results_df.to_csv(RESULTS_DIR / 'citation_results.csv', index=False)
    print(f"Saved results to results/citation_results.csv")

    return pl_results, regime_results, spectral_results


if __name__ == '__main__':
    results = main()
