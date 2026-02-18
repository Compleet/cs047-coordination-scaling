#!/usr/bin/env python3
"""
Attachment Kernel Estimation (Section 5.3)

Paper: "Coordination Scaling: Two Universality Classes Under Capacity Constraint"

Tests the sublinear preferential attachment claim: Pi(k) ~ k^{0.77 +/- 0.05}.
Uses the Newman (2001) / Jeong et al. (2003) method to estimate the attachment
kernel from temporal edge data in Bitcoin trust networks.

For each edge arrival, the method records the degree of the node receiving
the attachment and the exposure (node-time) at each degree level, then
computes Pi(k) = A_k / exposure_k. A log-log regression yields the
attachment exponent alpha.

Networks analyzed:
  - Bitcoin OTC (soc-sign-bitcoinotc)
  - Bitcoin Alpha (soc-sign-bitcoinalpha)

Outputs:
  - figures/attachment_kernel.png
  - results/attachment_kernel_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "trust_networks"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


def load_bitcoin_otc():
    """Load Bitcoin OTC edge list sorted by timestamp."""
    df = pd.read_csv(DATA_DIR / "soc-sign-bitcoinotc.csv", header=None,
                     names=['source', 'target', 'rating', 'time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def estimate_attachment_kernel_newman(edges_df, degree_type='in'):
    """
    Newman (2001) method for attachment kernel estimation.

    For each edge arrival:
    1. Record the current degree k of the node receiving the attachment.
    2. Record the current degree distribution (node counts at each degree).

    Then compute:
    Pi(k) = A_k / (N_k * T)

    where:
    - A_k = total attachments to nodes that had degree k at time of attachment
    - N_k * T = cumulative node-time exposure at degree k
    """
    node_degrees = {}
    degree_attachment_counts = {}
    degree_node_time = {}

    for idx, row in edges_df.iterrows():
        if degree_type == 'in':
            target_node = row['target']
        else:
            target_node = row['source']

        current_deg = node_degrees.get(target_node, 0)

        # Record attachment to this degree
        degree_attachment_counts[current_deg] = degree_attachment_counts.get(current_deg, 0) + 1

        # Record node-time exposure for all degrees at this timestep
        degree_counts = {}
        for node, deg in node_degrees.items():
            degree_counts[deg] = degree_counts.get(deg, 0) + 1

        for deg, count in degree_counts.items():
            degree_node_time[deg] = degree_node_time.get(deg, 0) + count

        # Update degree
        node_degrees[target_node] = current_deg + 1

    # Compute Pi(k) = A_k / exposure_k
    k_values = []
    pi_values = []

    for k in sorted(degree_attachment_counts.keys()):
        if k < 1:  # Skip degree 0 (new nodes)
            continue

        A_k = degree_attachment_counts.get(k, 0)
        exposure_k = degree_node_time.get(k, 0)

        if exposure_k > 0 and A_k > 5:  # Require minimum counts
            pi_k = A_k / exposure_k
            k_values.append(k)
            pi_values.append(pi_k)

    return np.array(k_values), np.array(pi_values)


def estimate_attachment_kernel_simple(edges_df, degree_type='in'):
    """
    Simplified attachment kernel estimate using quasi-static approximation.

    Counts attachments by degree and normalizes by time-averaged node counts
    at each degree level (approximated from the final degree distribution).
    """
    node_degrees = {}
    attachments_by_degree = {}

    for idx, row in edges_df.iterrows():
        if degree_type == 'in':
            target = row['target']
        else:
            target = row['source']

        current_deg = node_degrees.get(target, 0)

        if current_deg >= 1:
            attachments_by_degree[current_deg] = attachments_by_degree.get(current_deg, 0) + 1

        node_degrees[target] = current_deg + 1

    # Approximate average nodes at each degree from final distribution
    final_degree_dist = {}
    for node, deg in node_degrees.items():
        for d in range(1, deg + 1):
            final_degree_dist[d] = final_degree_dist.get(d, 0) + 1

    k_values = []
    pi_values = []

    for k in sorted(attachments_by_degree.keys()):
        if k < 1:
            continue

        A_k = attachments_by_degree[k]
        N_k = final_degree_dist.get(k, 1) / 2  # Rough time average

        if N_k > 0 and A_k > 5:
            pi_k = A_k / N_k
            k_values.append(k)
            pi_values.append(pi_k)

    return np.array(k_values), np.array(pi_values)


def fit_power_law(k, pi, k_min=1):
    """Fit power law Pi(k) ~ k^alpha via OLS on log-log data."""
    mask = (k >= k_min) & (pi > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan

    log_k = np.log(k[mask])
    log_pi = np.log(pi[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_pi)
    return slope, r_value**2, std_err


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Attachment Kernel Estimation (Section 5.3)")
    print("=" * 70)
    print("\nTests claim: Pi(k) ~ k^{0.77 +/- 0.05} (sublinear attachment)")
    print("Method: Newman (2001) / Jeong et al. (2003)")
    print("Pi(k) = (rate of attachment to degree-k nodes) / (exposure of degree-k nodes)\n")

    # Load data
    print("Loading Bitcoin OTC...")
    df_otc = load_bitcoin_otc()
    print(f"  Loaded {len(df_otc):,} edges")

    print("\nLoading Bitcoin Alpha...")
    df_alpha = pd.read_csv(DATA_DIR / "soc-sign-bitcoinalpha.csv", header=None,
                           names=['source', 'target', 'rating', 'time'])
    df_alpha = df_alpha.sort_values('time').reset_index(drop=True)
    print(f"  Loaded {len(df_alpha):,} edges")

    results = {}

    for name, df in [('Bitcoin_OTC', df_otc), ('Bitcoin_Alpha', df_alpha)]:
        print(f"\n{'=' * 50}")
        print(f"Analyzing {name}")
        print('=' * 50)

        for deg_type in ['in', 'out']:
            print(f"\n  {deg_type.upper()}-DEGREE:")

            # Use Newman method
            print("    Computing attachment kernel (Newman method)...")
            k, pi = estimate_attachment_kernel_newman(df, deg_type)

            if len(k) < 3:
                print("    Insufficient data")
                continue

            alpha, r2, std_err = fit_power_law(k, pi)

            print(f"    alpha = {alpha:.3f} +/- {std_err:.3f}")
            print(f"    R^2 = {r2:.4f}")
            print(f"    Data points: {len(k)}")

            # Simplified method for comparison
            k2, pi2 = estimate_attachment_kernel_simple(df, deg_type)
            alpha2, r2_2, _ = fit_power_law(k2, pi2)
            print(f"    (Simplified method: alpha = {alpha2:.3f})")

            results[f"{name}_{deg_type}"] = {
                'k': k, 'pi': pi, 'alpha': alpha, 'r2': r2,
                'std_err': std_err, 'k2': k2, 'pi2': pi2, 'alpha2': alpha2
            }

    # Create plot
    print("\nCreating plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (key, res) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]

        k, pi = res['k'], res['pi']
        mask = (k >= 1) & (pi > 0)

        ax.loglog(k[mask], pi[mask], 'bo', markersize=6, alpha=0.7, label='Data (Newman)')

        # Fit line
        if not np.isnan(res['alpha']):
            k_fit = np.logspace(np.log10(max(1, k[mask].min())), np.log10(k[mask].max()), 50)
            med_idx = len(k[mask]) // 2
            k_anchor = k[mask][med_idx]
            pi_anchor = pi[mask][med_idx]
            pi_fit = pi_anchor * (k_fit / k_anchor) ** res['alpha']
            ax.loglog(k_fit, pi_fit, 'r-', linewidth=2,
                      label=f'Fit: alpha={res["alpha"]:.2f}')

            # Reference lines
            pi_linear = pi_anchor * (k_fit / k_anchor) ** 1.0
            ax.loglog(k_fit, pi_linear, 'g--', alpha=0.5, label='Linear (alpha=1)')

            pi_paper = pi_anchor * (k_fit / k_anchor) ** 0.77
            ax.loglog(k_fit, pi_paper, 'orange', linestyle=':',
                      alpha=0.7, label='Paper (alpha=0.77)')

        ax.set_xlabel('Degree k')
        ax.set_ylabel('Pi(k)')
        ax.set_title(f'{key}\nalpha = {res["alpha"]:.3f}, R^2 = {res["r2"]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'attachment_kernel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {FIG_DIR / 'attachment_kernel.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary_rows = []
    for key, res in results.items():
        parts = key.rsplit('_', 1)
        summary_rows.append({
            'Network': parts[0],
            'Degree': parts[1],
            'alpha_newman': f"{res['alpha']:.3f}",
            'SE': f"{res['std_err']:.3f}",
            'R2': f"{res['r2']:.4f}",
            'alpha_simple': f"{res['alpha2']:.3f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(RESULTS_DIR / 'attachment_kernel_results.csv', index=False)

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION ASSESSMENT")
    print("=" * 70)

    paper_alpha = 0.77
    paper_ci = (0.72, 0.82)

    all_alphas = [r['alpha'] for r in results.values() if not np.isnan(r['alpha'])]
    mean_alpha = np.mean(all_alphas)

    for key, res in results.items():
        alpha = res['alpha']
        if np.isnan(alpha):
            continue
        within_ci = paper_ci[0] <= alpha <= paper_ci[1]
        sublinear = 0 < alpha < 1.0

        print(f"\n{key}:")
        print(f"  alpha = {alpha:.3f}")
        print(f"  Within paper's [0.72, 0.82]: {'YES' if within_ci else 'NO'}")

        if sublinear:
            print(f"  Sublinear (0 < alpha < 1): YES")
        elif alpha > 1.0:
            print(f"  Superlinear (alpha > 1)")
        elif alpha < 0:
            print(f"  Negative alpha")
        else:
            print(f"  Linear or near-linear")

    print(f"\n{'=' * 70}")
    print(f"Mean alpha = {mean_alpha:.3f}")

    if all(0.5 < a < 1.0 for a in all_alphas):
        print("Sublinear attachment confirmed across all networks.")
        if 0.72 <= mean_alpha <= 0.82:
            print(f"  Mean alpha matches paper prediction (0.77 +/- 0.05).")
    elif mean_alpha > 0:
        print("Results differ from paper prediction.")
    else:
        print("Methodology may need refinement.")

    return results


if __name__ == '__main__':
    results = main()
