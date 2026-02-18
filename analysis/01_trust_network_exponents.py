#!/usr/bin/env python3
"""
Trust Network Power-Law Exponent Analysis (Table 6)

Paper: "Coordination Scaling: Two Universality Classes Under Capacity Constraint"

Validates the power-law tail exponents reported in Table 6 for trust network
degree distributions using the Clauset-Shalizi-Newman (2009) methodology.
Compares fitted exponents against paper values and tests lognormal vs.
power-law preference for the distribution body.

Networks analyzed:
  - Epinions (soc-Epinions1): in-degree gamma ~ 1.705, out-degree gamma ~ 1.729
  - Bitcoin OTC (soc-sign-bitcoinotc): in gamma ~ 2.269, out gamma ~ 2.059
  - Bitcoin Alpha (soc-sign-bitcoinalpha): in gamma ~ 2.175, out gamma ~ 2.000

Outputs:
  - figures/degree_distributions.png
  - results/table6_comparison.csv
  - results/detailed_results.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_epinions():
    """Load Epinions trust network from SNAP edge-list format."""
    edges = []
    with open(DATA_DIR / "soc-Epinions1.txt") as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def load_bitcoin(filename):
    """Load Bitcoin trust network (OTC or Alpha) from CSV."""
    df = pd.read_csv(DATA_DIR / filename, header=None,
                     names=['source', 'target', 'rating', 'time'])
    G = nx.DiGraph()
    G.add_edges_from(zip(df['source'], df['target']))
    return G


def analyze_degree_distribution(G, name):
    """
    Analyze in-degree and out-degree distributions using the powerlaw package.

    Applies Clauset-Shalizi-Newman methodology: discrete MLE fitting with
    automatic xmin selection and likelihood-ratio tests against alternative
    distributions (exponential, lognormal, truncated power law).
    """
    results = {}

    for deg_type in ['in', 'out']:
        if deg_type == 'in':
            degrees = [d for n, d in G.in_degree() if d > 0]
        else:
            degrees = [d for n, d in G.out_degree() if d > 0]

        degrees = np.array(degrees)

        # Fit power law using Clauset-Shalizi-Newman methodology
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

        # Get power law parameters
        alpha = fit.power_law.alpha  # gamma in paper notation
        xmin = fit.power_law.xmin
        sigma = fit.power_law.sigma  # standard error

        # Compare distributions
        R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
        R_tpl, p_tpl = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)

        results[deg_type] = {
            'gamma': alpha,
            'xmin': xmin,
            'sigma': sigma,
            'n_samples': len(degrees),
            'n_above_xmin': np.sum(degrees >= xmin),
            'R_vs_exp': R_exp,
            'p_vs_exp': p_exp,
            'R_vs_lognormal': R_ln,
            'p_vs_lognormal': p_ln,
            'R_vs_truncated_pl': R_tpl,
            'p_vs_truncated_pl': p_tpl,
            'degrees': degrees,
            'fit': fit
        }

    return results


def create_comparison_table(all_results, paper_values):
    """Create table comparing measured exponents to paper Table 6 values."""
    rows = []
    for network, results in all_results.items():
        for deg_type in ['in', 'out']:
            r = results[deg_type]
            paper_key = f"{network}_{deg_type}"
            paper_gamma = paper_values.get(paper_key, np.nan)

            diff = abs(r['gamma'] - paper_gamma) if not np.isnan(paper_gamma) else np.nan
            within_threshold = diff <= 0.1 if not np.isnan(diff) else False

            # Determine which distribution is preferred
            if r['p_vs_lognormal'] < 0.05:
                if r['R_vs_lognormal'] > 0:
                    preferred = "Power Law"
                else:
                    preferred = "Lognormal"
            else:
                preferred = "Inconclusive"

            rows.append({
                'Network': network,
                'Degree': deg_type,
                'gamma_measured': f"{r['gamma']:.3f}",
                'sigma': f"{r['sigma']:.3f}",
                'gamma_paper': f"{paper_gamma:.3f}" if not np.isnan(paper_gamma) else "N/A",
                'delta': f"{diff:.3f}" if not np.isnan(diff) else "N/A",
                'within_0.1': 'Y' if within_threshold else 'N',
                'xmin': r['xmin'],
                'n_above_xmin': r['n_above_xmin'],
                'R_vs_lognormal': f"{r['R_vs_lognormal']:.3f}",
                'p_vs_lognormal': f"{r['p_vs_lognormal']:.4f}",
                'preferred_dist': preferred
            })

    return pd.DataFrame(rows)


def plot_distributions(all_results, output_dir):
    """Create log-log CCDF plots of degree distributions with fitted models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    networks = list(all_results.keys())

    for i, network in enumerate(networks):
        for j, deg_type in enumerate(['in', 'out']):
            ax = axes[j, i]
            r = all_results[network][deg_type]

            # Plot empirical CCDF
            fit = r['fit']
            fit.plot_ccdf(ax=ax, color='b', linewidth=2, label='Data')
            fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--',
                                     label=f'Power Law (gamma={r["gamma"]:.2f})')
            fit.lognormal.plot_ccdf(ax=ax, color='g', linestyle=':',
                                     label='Lognormal')

            ax.set_title(f'{network} {deg_type}-degree')
            ax.set_xlabel('Degree k')
            ax.set_ylabel('P(X >= k)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'degree_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved degree distribution plots to {output_dir / 'degree_distributions.png'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Trust Network Power-Law Exponents (Table 6)")
    print("=" * 70)

    # Paper's reported values (Table 6)
    paper_values = {
        'Epinions_in': 1.705,
        'Epinions_out': 1.729,
        'Bitcoin_OTC_in': 2.269,
        'Bitcoin_OTC_out': 2.059,
        'Bitcoin_Alpha_in': 2.175,
        'Bitcoin_Alpha_out': 2.000,
    }

    # Load networks
    print("\nLoading networks...")

    networks = {}
    print("  Loading Epinions...")
    networks['Epinions'] = load_epinions()
    print(f"    Nodes: {networks['Epinions'].number_of_nodes():,}, "
          f"Edges: {networks['Epinions'].number_of_edges():,}")

    print("  Loading Bitcoin OTC...")
    networks['Bitcoin_OTC'] = load_bitcoin("soc-sign-bitcoinotc.csv")
    print(f"    Nodes: {networks['Bitcoin_OTC'].number_of_nodes():,}, "
          f"Edges: {networks['Bitcoin_OTC'].number_of_edges():,}")

    print("  Loading Bitcoin Alpha...")
    networks['Bitcoin_Alpha'] = load_bitcoin("soc-sign-bitcoinalpha.csv")
    print(f"    Nodes: {networks['Bitcoin_Alpha'].number_of_nodes():,}, "
          f"Edges: {networks['Bitcoin_Alpha'].number_of_edges():,}")

    # Analyze each network
    print("\nAnalyzing degree distributions (this may take a few minutes)...")
    all_results = {}

    for name, G in networks.items():
        print(f"\n  Analyzing {name}...")
        all_results[name] = analyze_degree_distribution(G, name)
        print(f"    In-degree:  gamma = {all_results[name]['in']['gamma']:.3f} "
              f"+/- {all_results[name]['in']['sigma']:.3f}, "
              f"xmin = {all_results[name]['in']['xmin']}")
        print(f"    Out-degree: gamma = {all_results[name]['out']['gamma']:.3f} "
              f"+/- {all_results[name]['out']['sigma']:.3f}, "
              f"xmin = {all_results[name]['out']['xmin']}")

    # Create comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON (Table 6)")
    print("=" * 70)

    comparison_df = create_comparison_table(all_results, paper_values)
    print(comparison_df.to_string(index=False))

    # Save to CSV
    comparison_df.to_csv(RESULTS_DIR / 'table6_comparison.csv', index=False)
    print(f"\nSaved comparison table to {RESULTS_DIR / 'table6_comparison.csv'}")

    # Create plots
    plot_distributions(all_results, FIG_DIR)

    # Summary assessment
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    successes = comparison_df[comparison_df['within_0.1'] == 'Y'].shape[0]
    total = comparison_df.shape[0]
    print(f"\ngamma values within +/-0.1 of paper: {successes}/{total}")

    ln_preferred = comparison_df[comparison_df['preferred_dist'] == 'Lognormal'].shape[0]
    print(f"Networks where lognormal preferred over power law: {ln_preferred}/{total}")

    # Check for values outside expected ranges
    serious_issues = []
    for _, row in comparison_df.iterrows():
        gamma = float(row['gamma_measured'])
        network = row['Network']

        if 'Epinions' in network:
            if gamma < 1.5 or gamma > 2.5:
                serious_issues.append(
                    f"{network} {row['Degree']}: gamma = {gamma:.3f} outside [1.5, 2.5]")
        elif 'Bitcoin' in network:
            if gamma < 1.8 or gamma > 2.8:
                serious_issues.append(
                    f"{network} {row['Degree']}: gamma = {gamma:.3f} outside [1.8, 2.8]")

    if serious_issues:
        print("\nIssues detected:")
        for issue in serious_issues:
            print(f"    {issue}")
    else:
        print("\nAll gamma values within expected ranges.")

    # Save detailed results
    detailed_results = []
    for network, results in all_results.items():
        for deg_type in ['in', 'out']:
            r = results[deg_type]
            detailed_results.append({
                'network': network,
                'degree_type': deg_type,
                'gamma': r['gamma'],
                'sigma': r['sigma'],
                'xmin': r['xmin'],
                'n_samples': r['n_samples'],
                'n_above_xmin': r['n_above_xmin'],
                'R_vs_exponential': r['R_vs_exp'],
                'p_vs_exponential': r['p_vs_exp'],
                'R_vs_lognormal': r['R_vs_lognormal'],
                'p_vs_lognormal': r['p_vs_lognormal'],
                'R_vs_truncated_pl': r['R_vs_truncated_pl'],
                'p_vs_truncated_pl': r['p_vs_truncated_pl']
            })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(RESULTS_DIR / 'detailed_results.csv', index=False)
    print(f"Saved detailed results to {RESULTS_DIR / 'detailed_results.csv'}")

    return all_results, comparison_df


if __name__ == '__main__':
    all_results, comparison_df = main()
