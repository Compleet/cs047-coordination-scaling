#!/usr/bin/env python3
"""
Barabási-Albert Dynamic Null Model for ECP (SI Appendix J)

Paper: "Two Universality Classes of Coordination Scaling Under Capacity Constraint"

Tests whether spectral concentration rho_k in real trust networks exceeds
what preferential attachment growth alone would produce. Compares against
BA networks grown to matching size and average degree.

Key result: Trust networks show rho_k 2.5-2.8x higher than BA null
(z = 43-112), demonstrating that ECP reflects mesoscale coordination
structure beyond hub dominance.

Also includes temporal analysis: the gap widens as BTC OTC grows from
N=818 to N=5881 (z increases from ~18 to ~98).

Networks:
  - Bitcoin OTC (soc-sign-bitcoinotc)
  - Bitcoin Alpha (soc-sign-bitcoinalpha)

Outputs:
  - figures/ba_dynamic_null.png
  - results/ba_dynamic_null.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

N_BA_SAMPLES = 30
K_FRACS = [0.01, 0.05]


def load_bitcoin(filename):
    """Load Bitcoin trust network from CSV."""
    df = pd.read_csv(DATA_DIR / filename, header=None,
                     names=['source', 'target', 'rating', 'time'])
    G = nx.DiGraph()
    G.add_edges_from(zip(df['source'], df['target']))
    return G, df


def graph_to_sparse(G):
    """Convert graph to sparse adjacency matrix."""
    G2 = nx.convert_node_labels_to_integers(G)
    return nx.adjacency_matrix(G2).astype(float), G2.number_of_nodes()


def compute_rho_k(A_sparse, n, k_frac=0.01):
    """Compute spectral concentration rho_k."""
    if n < 20:
        return np.nan

    k = max(1, int(n * k_frac))
    k = min(k, n - 2)

    total = A_sparse.multiply(A_sparse).sum()
    if total == 0:
        return 0.0

    if n < 500:
        A_dense = A_sparse.toarray()
        _, s, _ = full_svd(A_dense, full_matrices=False)
        total = np.sum(s**2)
        top_k = np.sum(s[:k]**2)
    else:
        n_sv = min(k + 5, n - 2)
        try:
            _, s_top, _ = svds(A_sparse.astype(float), k=n_sv)
            s_top = np.sort(s_top)[::-1]
            top_k = np.sum(s_top[:k]**2)
        except Exception:
            return np.nan

    return top_k / total if total > 0 else 0.0


def generate_ba_digraph(n, m):
    """Generate directed BA graph by randomly orienting undirected BA edges."""
    m = max(1, min(m, n - 1))
    G_und = nx.barabasi_albert_graph(n, m)
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(range(n))
    rands = np.random.random(G_und.number_of_edges())
    for i, (u, v) in enumerate(G_und.edges()):
        if rands[i] < 0.5:
            G_dir.add_edge(u, v)
        else:
            G_dir.add_edge(v, u)
    return G_dir


def compute_ba_null_rho(n, m_ba, k_frac, n_samples=30):
    """Compute rho_k for n_samples BA graphs."""
    rho_values = []
    for i in range(n_samples):
        G_ba = generate_ba_digraph(n, m_ba)
        A_ba, n_ba = graph_to_sparse(G_ba)
        rho = compute_rho_k(A_ba, n_ba, k_frac)
        if not np.isnan(rho):
            rho_values.append(rho)
    return np.array(rho_values)


def main():
    np.random.seed(42)

    print("=" * 70)
    print("BA Dynamic Null Model for ECP (SI Appendix J)")
    print("=" * 70)

    networks = {
        'Bitcoin OTC': 'soc-sign-bitcoinotc.csv',
        'Bitcoin Alpha': 'soc-sign-bitcoinalpha.csv',
    }

    all_results = []

    # ---- PART 1: Full network analysis ----
    for name, filename in networks.items():
        G, df = load_bitcoin(filename)
        n = G.number_of_nodes()
        m_edges = G.number_of_edges()
        m_ba = max(1, round(m_edges / n))

        print(f"\n{name}: N={n}, M={m_edges}, m_ba={m_ba}")

        A_real, n_real = graph_to_sparse(G)

        for k_frac in K_FRACS:
            k = max(1, int(n * k_frac))
            rho_real = compute_rho_k(A_real, n_real, k_frac)
            rho_ba = compute_ba_null_rho(n, m_ba, k_frac, N_BA_SAMPLES)

            if len(rho_ba) > 0:
                ba_mean = np.mean(rho_ba)
                ba_std = np.std(rho_ba)
                z = (rho_real - ba_mean) / ba_std if ba_std > 0 else np.inf
                ratio = rho_real / ba_mean if ba_mean > 0 else np.inf

                print(f"  k={k_frac}: rho_real={rho_real:.4f}, "
                      f"rho_BA={ba_mean:.4f}±{ba_std:.4f}, "
                      f"z={z:.1f}, ratio={ratio:.2f}x")

                all_results.append({
                    'network': name, 'snapshot': 'full',
                    'k_frac': k_frac, 'k': k,
                    'n_nodes': n, 'n_edges': m_edges,
                    'rho_real': rho_real,
                    'rho_ba_mean': ba_mean, 'rho_ba_std': ba_std,
                    'z_score': z, 'ratio': ratio,
                })

    # ---- PART 2: Temporal analysis (BTC OTC) ----
    print(f"\n{'='*70}")
    print("TEMPORAL ANALYSIS: Bitcoin OTC")
    print(f"{'='*70}")

    G_otc, df_otc = load_bitcoin('soc-sign-bitcoinotc.csv')
    df_otc = df_otc.sort_values('time').reset_index(drop=True)

    n_snapshots = 10
    quantiles = np.linspace(0.1, 1.0, n_snapshots)
    time_points = [df_otc['time'].quantile(q) for q in quantiles]

    for i, t_cut in enumerate(time_points):
        df_snap = df_otc[df_otc['time'] <= t_cut]
        G = nx.DiGraph()
        G.add_edges_from(zip(df_snap['source'].astype(int),
                             df_snap['target'].astype(int)))

        n = G.number_of_nodes()
        m_edges = G.number_of_edges()
        m_ba = max(1, round(m_edges / n))

        if n < 50:
            continue

        A_real, n_real = graph_to_sparse(G)
        rho_real = compute_rho_k(A_real, n_real, 0.01)

        rho_ba = compute_ba_null_rho(n, m_ba, 0.01, 20)

        if len(rho_ba) > 0 and not np.isnan(rho_real):
            ba_mean = np.mean(rho_ba)
            ba_std = np.std(rho_ba)
            z = (rho_real - ba_mean) / ba_std if ba_std > 0 else np.inf

            print(f"  Snap {i+1}: N={n}, rho={rho_real:.4f}, "
                  f"BA={ba_mean:.4f}, z={z:.1f}")

            all_results.append({
                'network': 'Bitcoin OTC',
                'snapshot': f'temporal_{i+1}',
                'k_frac': 0.01,
                'k': max(1, int(n * 0.01)),
                'n_nodes': n, 'n_edges': m_edges,
                'rho_real': rho_real,
                'rho_ba_mean': ba_mean, 'rho_ba_std': ba_std,
                'z_score': z,
                'ratio': rho_real / ba_mean if ba_mean > 0 else np.inf,
            })

    # ---- Save ----
    df_results = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_DIR / 'ba_dynamic_null.csv', index=False)
    print(f"\nSaved results to results/ba_dynamic_null.csv")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ECP vs Barabási-Albert Dynamic Null (SI Appendix J)',
                 fontsize=13, fontweight='bold')

    # Panel A: Full network z-scores
    ax = axes[0]
    full = df_results[df_results['snapshot'] == 'full']
    labels = [f"{r['network']}\nk={r['k_frac']}" for _, r in full.iterrows()]
    z_scores = full['z_score'].values
    colors = ['#c0392b' if z > 10 else '#e67e22' for z in z_scores]
    bars = ax.bar(range(len(labels)), z_scores, color=colors, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Z-score (real vs BA null)')
    ax.set_title('Full Network: Z-scores')
    ax.axhline(2, color='gray', ls='--', lw=0.8)
    for bar, z in zip(bars, z_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{z:.0f}', ha='center', fontsize=9, fontweight='bold')

    # Panel B: Temporal
    ax = axes[1]
    temp = df_results[df_results['snapshot'].str.startswith('temporal')]
    if len(temp) > 0:
        temp = temp.sort_values('n_nodes')
        ax.plot(temp['n_nodes'], temp['rho_real'], 'o-', color='#c0392b',
                lw=2, ms=5, label='Real network')
        ax.plot(temp['n_nodes'], temp['rho_ba_mean'], 's--', color='#3498db',
                lw=2, ms=5, label='BA null (mean)')
        ax.fill_between(temp['n_nodes'],
                        temp['rho_ba_mean'] - temp['rho_ba_std'],
                        temp['rho_ba_mean'] + temp['rho_ba_std'],
                        alpha=0.2, color='#3498db')
        ax.set_xlabel('Network size N')
        ax.set_ylabel(r'$\rho_k$ (k = 1% of N)')
        ax.set_title('Temporal: Bitcoin OTC')
        ax.legend(fontsize=9)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / 'ba_dynamic_null.png', dpi=150, bbox_inches='tight')
    print(f"Saved figure to figures/ba_dynamic_null.png")
    plt.close()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    full_1pct = full[full['k_frac'] == 0.01]
    for _, r in full_1pct.iterrows():
        print(f"  {r['network']}: rho_real/rho_BA = {r['ratio']:.1f}x, z = {r['z_score']:.0f}")


if __name__ == '__main__':
    main()
