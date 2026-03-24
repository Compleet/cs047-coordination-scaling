#!/usr/bin/env python3
"""
Granger Causality Analysis: γ vs ρ_k Lead/Lag (SI Appendix K)

Paper: "Two Universality Classes of Coordination Scaling Under Capacity Constraint"

Tests temporal ordering between degree distribution tail exponent γ and
spectral concentration ρ_k using cumulative snapshots of Bitcoin OTC.

Key result: γ temporally precedes ρ_k (Granger F=22.7, p=0.0001),
but PA dynamic null shows ρ_k far exceeds what γ alone produces
(z=43-112). Interpretation: degree evolution provides initial perturbation;
coordination topology amplifies collapse.

Network: Bitcoin OTC (25 cumulative temporal snapshots)

Outputs:
  - figures/granger_ecp.png
  - results/granger_ecp.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd
import powerlaw
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_bitcoin_otc():
    """Load Bitcoin OTC with timestamps."""
    df = pd.read_csv(DATA_DIR / "soc-sign-bitcoinotc.csv", header=None,
                     names=['source', 'target', 'rating', 'time'])
    return df.sort_values('time').reset_index(drop=True)


def compute_rho_k(G, k_frac=0.01):
    """Compute spectral concentration."""
    G2 = nx.convert_node_labels_to_integers(G)
    n = G2.number_of_nodes()
    if n < 50:
        return np.nan

    A = nx.adjacency_matrix(G2).astype(float)
    k = max(1, int(n * k_frac))
    k = min(k, n - 2)

    total = A.multiply(A).sum()
    if total == 0:
        return 0.0

    if n < 500:
        _, s, _ = full_svd(A.toarray(), full_matrices=False)
        total = np.sum(s**2)
        top_k = np.sum(s[:k]**2)
    else:
        n_sv = min(k + 5, n - 2)
        try:
            _, s_top, _ = svds(A, k=n_sv)
            s_top = np.sort(s_top)[::-1]
            top_k = np.sum(s_top[:k]**2)
        except Exception:
            return np.nan

    return top_k / total


def compute_gamma(G):
    """Compute power-law tail exponent of in-degree distribution."""
    degrees = [d for _, d in G.in_degree() if d > 0]
    if len(degrees) < 50:
        return np.nan
    try:
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        return fit.alpha
    except Exception:
        return np.nan


def granger_test(x, y, max_lag=3):
    """Simple Granger causality: does x help predict y beyond y's own lags?

    Returns F-statistic and p-value for each lag.
    """
    from scipy.stats import f as f_dist

    results = []
    n = len(x)

    for lag in range(1, max_lag + 1):
        if n - lag < lag + 3:
            continue

        # Restricted model: y_t ~ y_{t-1} + ... + y_{t-lag}
        Y = y[lag:]
        X_restricted = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])

        # Unrestricted: y_t ~ y_{t-1}...y_{t-lag} + x_{t-1}...x_{t-lag}
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag-i-1:n-i-1] for i in range(lag)]
        ])

        # Add intercept
        X_r = np.column_stack([np.ones(len(Y)), X_restricted])
        X_u = np.column_stack([np.ones(len(Y)), X_unrestricted])

        # OLS
        try:
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        ssr_r = np.sum((Y - X_r @ beta_r)**2)
        ssr_u = np.sum((Y - X_u @ beta_u)**2)

        df1 = lag  # additional parameters
        df2 = len(Y) - X_u.shape[1]

        if df2 <= 0 or ssr_u <= 0:
            continue

        F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p = 1 - f_dist.cdf(F, df1, df2)

        results.append({'lag': lag, 'F': F, 'p': p})

    return results


def main():
    print("=" * 70)
    print("Granger Causality: γ vs ρ_k (SI Appendix K)")
    print("=" * 70)

    df = load_bitcoin_otc()

    # Create 25 cumulative snapshots
    n_snapshots = 25
    quantiles = np.linspace(0.04, 1.0, n_snapshots)
    time_points = [df['time'].quantile(q) for q in quantiles]

    snapshots = []
    for i, t_cut in enumerate(time_points):
        df_snap = df[df['time'] <= t_cut]
        G = nx.DiGraph()
        G.add_edges_from(zip(df_snap['source'].astype(int),
                             df_snap['target'].astype(int)))

        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n < 100:
            continue

        rho_1 = compute_rho_k(G, 0.01)
        rho_5 = compute_rho_k(G, 0.05)
        gamma = compute_gamma(G)

        print(f"  Snap {i+1:2d}: N={n:5d}, γ={gamma:.3f}, "
              f"ρ_1%={rho_1:.4f}, ρ_5%={rho_5:.4f}")

        snapshots.append({
            'snapshot': i + 1, 'n_nodes': n, 'n_edges': m,
            'gamma': gamma, 'rho_1pct': rho_1, 'rho_5pct': rho_5,
        })

    df_snap = pd.DataFrame(snapshots)

    # Granger tests
    gamma_arr = df_snap['gamma'].values
    rho1_arr = df_snap['rho_1pct'].values
    rho5_arr = df_snap['rho_5pct'].values

    print(f"\n{'='*70}")
    print("GRANGER CAUSALITY TESTS")
    print(f"{'='*70}")

    print("\nγ → ρ_k (does degree exponent predict spectral concentration?):")
    g_to_r = granger_test(gamma_arr, rho1_arr, max_lag=3)
    for r in g_to_r:
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
        print(f"  Lag {r['lag']}: F = {r['F']:.1f}, p = {r['p']:.4f} {sig}")

    print("\nρ_k → γ (reverse direction):")
    r_to_g = granger_test(rho1_arr, gamma_arr, max_lag=3)
    for r in r_to_g:
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
        print(f"  Lag {r['lag']}: F = {r['F']:.1f}, p = {r['p']:.4f} {sig}")

    # Cross-correlation
    from scipy.stats import pearsonr
    print(f"\nCross-correlations (γ vs ρ_1%):")
    for lag in range(-3, 4):
        if lag < 0:
            r, p = pearsonr(gamma_arr[:lag], rho1_arr[-lag:])
        elif lag > 0:
            r, p = pearsonr(gamma_arr[lag:], rho1_arr[:-lag])
        else:
            r, p = pearsonr(gamma_arr, rho1_arr)
        print(f"  Lag {lag:+d}: r = {r:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_snap.to_csv(RESULTS_DIR / 'granger_ecp.csv', index=False)

    # Save Granger test summary
    granger_summary = {
        'gamma_to_rho': g_to_r,
        'rho_to_gamma': r_to_g,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Granger Analysis: γ vs ρ_k (SI Appendix K)',
                 fontsize=13, fontweight='bold')

    # Panel A: Time series
    ax = axes[0]
    ax2 = ax.twinx()
    ln1 = ax.plot(df_snap['n_nodes'], df_snap['rho_1pct'], 'o-',
                  color='#c0392b', lw=2, ms=4, label=r'$\rho_k$ (k=1%)')
    ln2 = ax2.plot(df_snap['n_nodes'], df_snap['gamma'], 's--',
                   color='#2c3e50', lw=2, ms=4, label=r'$\gamma$ (in-degree)')
    ax.set_xlabel('Network size N')
    ax.set_ylabel(r'$\rho_k$', color='#c0392b')
    ax2.set_ylabel(r'$\gamma$', color='#2c3e50')
    ax.set_title('Temporal Co-evolution')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=9)

    # Panel B: Cross-correlation
    ax = axes[1]
    lags = range(-3, 4)
    cc = []
    for lag in lags:
        if lag < 0:
            r, _ = pearsonr(gamma_arr[:lag], rho1_arr[-lag:])
        elif lag > 0:
            r, _ = pearsonr(gamma_arr[lag:], rho1_arr[:-lag])
        else:
            r, _ = pearsonr(gamma_arr, rho1_arr)
        cc.append(r)
    ax.bar(list(lags), cc, color=['#3498db' if l < 0 else '#e67e22' for l in lags],
           edgecolor='black', alpha=0.7)
    ax.set_xlabel('Lag (negative = γ leads)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('γ–ρ_k Cross-Correlation')
    ax.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / 'granger_ecp.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to figures/granger_ecp.png and results/granger_ecp.csv")
    plt.close()

    # Verdict
    print(f"\n{'='*70}")
    best_fwd = min(g_to_r, key=lambda x: x['p']) if g_to_r else None
    best_rev = min(r_to_g, key=lambda x: x['p']) if r_to_g else None
    if best_fwd and best_fwd['p'] < 0.01:
        print(f"γ → ρ_k: SIGNIFICANT (F={best_fwd['F']:.1f}, p={best_fwd['p']:.4f})")
    else:
        print(f"γ → ρ_k: not significant")
    if best_rev and best_rev['p'] < 0.05:
        print(f"ρ_k → γ: marginally significant (F={best_rev['F']:.1f}, p={best_rev['p']:.4f})")
    else:
        print(f"ρ_k → γ: not significant")


if __name__ == '__main__':
    main()
