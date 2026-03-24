#!/usr/bin/env python3
"""
CS-047 CSD Robustness Analysis — Phase 2: Full Robustness Suite
================================================================

Runs six robustness analyses on repo_summary.csv to defend the
critical-slowing-down claim against PNAS Statistical Review.

Usage:
    python 16b_csd_robustness.py

Requires: repo_summary.csv from Phase 1 (16a_extract_repos.py)

Analyses:
    A. Binning sensitivity (original bins, equal-count, equal-width, fine-grained)
    B. Sliding window variance (continuous, no bins)
    C. Piecewise regression (data-driven breakpoint detection)
    D. Mixed-effects model (random intercepts)
    E. Bootstrap confidence intervals on variance peak
    F. Confounder proxy check (repo size as rough control)

Output: results/csd_robustness/
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import linregress, levene, spearmanr
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "csd_robustness"
FIG_DIR = RESULTS_DIR
REPO_CSV = RESULTS_DIR / "repo_summary.csv"


def load_data():
    """Load per-repo summary."""
    if not REPO_CSV.exists():
        print(f"ERROR: {REPO_CSV} not found.")
        print("Run 16a_extract_repos.py first to extract per-repo data.")
        raise SystemExit(1)
    
    df = pd.read_csv(REPO_CSV)
    print(f"Loaded {len(df):,} repos")
    print(f"  N range: {df['n_contributors'].min()} – {df['n_contributors'].max()}")
    return df


def compute_beta_in_bin(df_bin, min_repos=20):
    """Compute OLS beta and its variance via bootstrap within a bin."""
    if len(df_bin) < min_repos:
        return np.nan, np.nan, np.nan, len(df_bin)
    
    log_n = np.log(df_bin['n_contributors'].values)
    log_prs = np.log(df_bin['n_prs'].values.clip(min=1))
    
    slope, intercept, r, p, se = linregress(log_n, log_prs)
    return slope, se, r**2, len(df_bin)


def bootstrap_beta_variance(df_bin, n_boot=1000, min_repos=15):
    """Bootstrap the variance of beta within a bin."""
    if len(df_bin) < min_repos:
        return np.nan, np.nan, np.nan, np.nan
    
    log_n = np.log(df_bin['n_contributors'].values)
    log_prs = np.log(df_bin['n_prs'].values.clip(min=1))
    n = len(log_n)
    
    betas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            slope, _, _, _, _ = linregress(log_n[idx], log_prs[idx])
            betas.append(slope)
        except:
            pass
    
    betas = np.array(betas)
    if len(betas) < 100:
        return np.nan, np.nan, np.nan, np.nan
    
    return (np.var(betas), 
            np.percentile(betas, 2.5), 
            np.percentile(betas, 97.5),
            np.mean(betas))


# =========================================================================
# ANALYSIS A: Binning Sensitivity
# =========================================================================
def analysis_a_binning_sensitivity(df):
    """Test variance peak under 4 different binning strategies."""
    print("\n" + "="*60)
    print("ANALYSIS A: Binning Sensitivity")
    print("="*60)
    
    results = {}
    
    # Strategy 1: Original paper bins
    original_edges = [5, 10, 20, 50, 100, 200, 500]
    original_labels = ['5-10', '10-20', '20-50', '50-100', '100-200', '200-500']
    
    # Strategy 2: Equal-count quantile bins (same number of repos per bin)
    n_quantile_bins = 6
    quantiles = np.linspace(0, 1, n_quantile_bins + 1)
    quantile_edges = df['n_contributors'].quantile(quantiles).values
    quantile_edges[0] = df['n_contributors'].min() - 0.5
    quantile_edges[-1] = df['n_contributors'].max() + 0.5
    
    # Strategy 3: Equal-width log-space bins
    log_min = np.log10(df['n_contributors'].min())
    log_max = np.log10(df['n_contributors'].max())
    logspace_edges = np.logspace(log_min, log_max, 8)
    
    # Strategy 4: Fine-grained (12 bins)
    fine_quantiles = np.linspace(0, 1, 13)
    fine_edges = df['n_contributors'].quantile(fine_quantiles).values
    fine_edges[0] = df['n_contributors'].min() - 0.5
    fine_edges[-1] = df['n_contributors'].max() + 0.5
    
    strategies = {
        'Original (paper)': original_edges,
        'Equal-count (6 bins)': quantile_edges,
        'Log-space (7 bins)': logspace_edges,
        'Fine-grained (12 bins)': fine_edges,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analysis A: Binning Sensitivity of Variance Peak', 
                 fontsize=13, fontweight='bold')
    
    for idx, (name, edges) in enumerate(strategies.items()):
        ax = axes[idx // 2][idx % 2]
        edges = np.unique(np.sort(edges))
        
        bin_results = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            mask = (df['n_contributors'] >= lo) & (df['n_contributors'] < hi)
            df_bin = df[mask]
            
            if len(df_bin) < 15:
                continue
            
            var, ci_lo, ci_hi, mean_beta = bootstrap_beta_variance(df_bin)
            midpoint = np.sqrt(lo * hi)  # geometric mean
            
            bin_results.append({
                'lo': lo, 'hi': hi, 'midpoint': midpoint,
                'n_repos': len(df_bin),
                'beta_mean': mean_beta,
                'beta_var': var,
                'beta_ci_lo': ci_lo,
                'beta_ci_hi': ci_hi,
            })
        
        if not bin_results:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue
        
        br = pd.DataFrame(bin_results)
        
        # Find peak
        if br['beta_var'].notna().any():
            peak_idx = br['beta_var'].idxmax()
            peak_row = br.loc[peak_idx]
            baseline = br['beta_var'].iloc[0] if br['beta_var'].iloc[0] > 0 else br['beta_var'].min()
            peak_ratio = peak_row['beta_var'] / baseline if baseline > 0 else np.nan
        else:
            peak_ratio = np.nan
        
        # Plot
        colors = ['#2ecc71' if b < 1 else '#e74c3c' for b in br['beta_mean'].fillna(1)]
        bars = ax.bar(range(len(br)), br['beta_var'], color=colors, 
                      edgecolor='black', alpha=0.8)
        
        labels = [f"{int(r['lo'])}-{int(r['hi'])}\nn={r['n_repos']}" 
                  for _, r in br.iterrows()]
        ax.set_xticks(range(len(br)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Var(β)')
        ax.set_title(f'{name}\nPeak ratio: {peak_ratio:.0f}×' if not np.isnan(peak_ratio) 
                     else name)
        
        results[name] = {
            'bins': len(br),
            'peak_ratio': float(peak_ratio) if not np.isnan(peak_ratio) else None,
            'peak_bin': f"{peak_row['lo']:.0f}-{peak_row['hi']:.0f}" if not np.isnan(peak_ratio) else None,
            'peak_n': int(peak_row['n_repos']) if not np.isnan(peak_ratio) else None,
        }
        
        print(f"\n  {name}:")
        print(f"    Bins: {len(br)}")
        print(f"    Peak: {results[name]['peak_bin']} (n={results[name]['peak_n']}, "
              f"ratio={peak_ratio:.0f}×)" if not np.isnan(peak_ratio) else "    No peak found")
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_binning_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: fig_binning_sensitivity.png")
    
    return results


# =========================================================================
# ANALYSIS B: Sliding Window Variance
# =========================================================================
def analysis_b_sliding_window(df):
    """Continuous variance estimation using overlapping log-space windows."""
    print("\n" + "="*60)
    print("ANALYSIS B: Sliding Window Variance (No Bins)")
    print("="*60)
    
    log_n = np.log(df['n_contributors'].values)
    log_prs = np.log(df['n_prs'].values.clip(min=1))
    
    # Window centers from log(5) to log(500), width = 0.8 in log-space (~2.2x range)
    centers = np.linspace(np.log(8), np.log(400), 40)
    half_width = 0.4  # each window spans ~ [center/1.5, center*1.5]
    
    window_results = []
    for c in centers:
        mask = (log_n >= c - half_width) & (log_n <= c + half_width)
        n_in_window = mask.sum()
        
        if n_in_window < 30:
            continue
        
        # Bootstrap variance of beta in this window
        betas = []
        for _ in range(500):
            idx = np.random.choice(np.where(mask)[0], n_in_window, replace=True)
            try:
                s, _, _, _, _ = linregress(log_n[idx], log_prs[idx])
                betas.append(s)
            except:
                pass
        
        if len(betas) < 100:
            continue
        
        betas = np.array(betas)
        window_results.append({
            'center_n': np.exp(c),
            'center_log_n': c,
            'n_repos': n_in_window,
            'beta_mean': np.mean(betas),
            'beta_var': np.var(betas),
            'beta_sd': np.std(betas),
        })
    
    wr = pd.DataFrame(window_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Analysis B: Sliding Window Variance (No Arbitrary Bins)',
                 fontsize=13, fontweight='bold')
    
    # Panel A: Variance
    ax = axes[0]
    ax.plot(wr['center_n'], wr['beta_var'], 'o-', color='#8e44ad', lw=2, ms=4)
    ax.set_xscale('log')
    ax.set_xlabel('Team size N (window center)')
    ax.set_ylabel('Var(β) in window')
    ax.set_title('Variance of β (sliding window)')
    ax.axvspan(80, 250, alpha=0.1, color='orange', label='Expected peak zone')
    ax.legend()
    
    # Panel B: Mean beta
    ax = axes[1]
    ax.plot(wr['center_n'], wr['beta_mean'], 'o-', color='#2c3e50', lw=2, ms=4)
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='β = 1')
    ax.set_xscale('log')
    ax.set_xlabel('Team size N (window center)')
    ax.set_ylabel('Mean β in window')
    ax.set_title('Mean β (sliding window)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_sliding_window.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find peak
    if len(wr) > 0 and wr['beta_var'].notna().any():
        peak = wr.loc[wr['beta_var'].idxmax()]
        baseline = wr['beta_var'].iloc[0]
        ratio = peak['beta_var'] / baseline if baseline > 0 else np.nan
        
        print(f"\n  Peak variance at N ≈ {peak['center_n']:.0f}")
        print(f"  Peak ratio: {ratio:.0f}× baseline")
        print(f"  β at peak: {peak['beta_mean']:.3f}")
        print(f"  β crosses 1.0 at N ≈ {wr.loc[(wr['beta_mean'] - 1.0).abs().idxmin(), 'center_n']:.0f}")
        
        result = {
            'peak_n': float(peak['center_n']),
            'peak_ratio': float(ratio),
            'beta_at_peak': float(peak['beta_mean']),
            'crossing_n': float(wr.loc[(wr['beta_mean'] - 1.0).abs().idxmin(), 'center_n']),
            'window_half_width_log': half_width,
        }
    else:
        result = {'peak_n': None}
    
    print(f"  Saved: fig_sliding_window.png")
    return result


# =========================================================================
# ANALYSIS C: Piecewise Regression (Breakpoint Detection)
# =========================================================================
def analysis_c_piecewise(df):
    """Fit piecewise linear model in log-log space; estimate breakpoint."""
    print("\n" + "="*60)
    print("ANALYSIS C: Piecewise Regression (Breakpoint Detection)")
    print("="*60)
    
    log_n = np.log(df['n_contributors'].values)
    log_prs = np.log(df['n_prs'].values.clip(min=1))
    
    def piecewise_sse(breakpoint):
        """Sum of squared errors for piecewise linear fit."""
        mask_lo = log_n <= breakpoint
        mask_hi = log_n > breakpoint
        
        if mask_lo.sum() < 50 or mask_hi.sum() < 50:
            return 1e18
        
        # Fit each segment
        s1, i1, _, _, _ = linregress(log_n[mask_lo], log_prs[mask_lo])
        s2, i2, _, _, _ = linregress(log_n[mask_hi], log_prs[mask_hi])
        
        resid_lo = log_prs[mask_lo] - (s1 * log_n[mask_lo] + i1)
        resid_hi = log_prs[mask_hi] - (s2 * log_n[mask_hi] + i2)
        
        return np.sum(resid_lo**2) + np.sum(resid_hi**2)
    
    # Grid search for breakpoint
    candidates = np.linspace(np.log(15), np.log(400), 100)
    sse_values = [piecewise_sse(c) for c in candidates]
    
    best_bp = candidates[np.argmin(sse_values)]
    best_n = np.exp(best_bp)
    
    # Fit segments at best breakpoint
    mask_lo = log_n <= best_bp
    mask_hi = log_n > best_bp
    s1, i1, r1, _, se1 = linregress(log_n[mask_lo], log_prs[mask_lo])
    s2, i2, r2, _, se2 = linregress(log_n[mask_hi], log_prs[mask_hi])
    
    # Compare with single linear fit
    s_all, i_all, r_all, _, _ = linregress(log_n, log_prs)
    sse_single = np.sum((log_prs - (s_all * log_n + i_all))**2)
    sse_piece = piecewise_sse(best_bp)
    
    # F-test for improvement (2 extra parameters)
    n_obs = len(log_n)
    df1 = 2  # extra params
    df2 = n_obs - 4  # piecewise has 4 params
    f_stat = ((sse_single - sse_piece) / df1) / (sse_piece / df2)
    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    
    print(f"\n  Best breakpoint: N ≈ {best_n:.0f} (log N = {best_bp:.2f})")
    print(f"  Left segment:  β = {s1:.3f} ± {se1:.3f} (R² = {r1**2:.3f}, n = {mask_lo.sum()})")
    print(f"  Right segment: β = {s2:.3f} ± {se2:.3f} (R² = {r2**2:.3f}, n = {mask_hi.sum()})")
    print(f"  Single-line:   β = {s_all:.3f} (R² = {r_all**2:.3f})")
    print(f"  F-test: F = {f_stat:.1f}, p = {p_value:.2e}")
    print(f"  Piecewise {'significantly' if p_value < 0.001 else 'does not significantly'} "
          f"improve fit over single line")
    
    # Bootstrap the breakpoint
    bp_boots = []
    for _ in range(500):
        idx = np.random.choice(n_obs, n_obs, replace=True)
        ln_b, lp_b = log_n[idx], log_prs[idx]
        
        def sse_boot(bp):
            m_lo = ln_b <= bp
            m_hi = ln_b > bp
            if m_lo.sum() < 30 or m_hi.sum() < 30:
                return 1e18
            s1b, i1b, _, _, _ = linregress(ln_b[m_lo], lp_b[m_lo])
            s2b, i2b, _, _, _ = linregress(ln_b[m_hi], lp_b[m_hi])
            r_lo = lp_b[m_lo] - (s1b * ln_b[m_lo] + i1b)
            r_hi = lp_b[m_hi] - (s2b * ln_b[m_hi] + i2b)
            return np.sum(r_lo**2) + np.sum(r_hi**2)
        
        bp_sse = [sse_boot(c) for c in candidates[::5]]
        bp_boots.append(np.exp(candidates[::5][np.argmin(bp_sse)]))
    
    bp_boots = np.array(bp_boots)
    bp_ci = np.percentile(bp_boots, [2.5, 97.5])
    
    print(f"  Breakpoint 95% CI: [{bp_ci[0]:.0f}, {bp_ci[1]:.0f}]")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Analysis C: Piecewise Regression with Breakpoint Detection',
                 fontsize=13, fontweight='bold')
    
    ax = axes[0]
    ax.scatter(log_n, log_prs, s=1, alpha=0.05, color='gray')
    x_lo = np.linspace(log_n.min(), best_bp, 50)
    x_hi = np.linspace(best_bp, log_n.max(), 50)
    ax.plot(x_lo, s1 * x_lo + i1, color='#e74c3c', lw=2.5, 
            label=f'β={s1:.2f} (N<{best_n:.0f})')
    ax.plot(x_hi, s2 * x_hi + i2, color='#2ecc71', lw=2.5,
            label=f'β={s2:.2f} (N>{best_n:.0f})')
    ax.axvline(best_bp, color='orange', ls='--', lw=1.5, label=f'Breakpoint N≈{best_n:.0f}')
    ax.set_xlabel('log(N)')
    ax.set_ylabel('log(PRs)')
    ax.set_title('Piecewise Linear Fit')
    ax.legend(fontsize=9)
    
    ax = axes[1]
    ax.hist(bp_boots, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(best_n, color='red', lw=2, label=f'Best: N={best_n:.0f}')
    ax.axvline(bp_ci[0], color='orange', ls='--', label=f'95% CI: [{bp_ci[0]:.0f}, {bp_ci[1]:.0f}]')
    ax.axvline(bp_ci[1], color='orange', ls='--')
    ax.set_xlabel('Breakpoint N')
    ax.set_ylabel('Bootstrap count')
    ax.set_title('Breakpoint Uncertainty (500 bootstraps)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_piecewise_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_piecewise_regression.png")
    
    return {
        'breakpoint_n': float(best_n),
        'breakpoint_ci': [float(bp_ci[0]), float(bp_ci[1])],
        'beta_left': float(s1),
        'beta_right': float(s2),
        'f_stat': float(f_stat),
        'p_value': float(p_value),
    }


# =========================================================================
# ANALYSIS D: Mixed-Effects Model
# =========================================================================
def analysis_d_mixed_effects(df):
    """Hierarchical model: random intercepts to account for repo heterogeneity."""
    print("\n" + "="*60)
    print("ANALYSIS D: Mixed-Effects / Hierarchical Model")
    print("="*60)
    
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not available. pip install statsmodels")
        return {'error': 'statsmodels not installed'}
    
    # Create size category as grouping variable
    df = df.copy()
    df['log_n'] = np.log(df['n_contributors'])
    df['log_prs'] = np.log(df['n_prs'].clip(lower=1))
    
    # Size quintiles as groups
    df['size_group'] = pd.qcut(df['n_contributors'], 5, labels=False, duplicates='drop')
    
    # Model 1: Simple OLS (baseline)
    import statsmodels.api as sm
    X = sm.add_constant(df['log_n'])
    ols = sm.OLS(df['log_prs'], X).fit()
    
    print(f"\n  OLS (all repos):")
    print(f"    β = {ols.params.iloc[1]:.4f} ± {ols.bse.iloc[1]:.4f}")
    print(f"    R² = {ols.rsquared:.4f}")
    
    # Model 2: OLS with interaction (size_group × log_n)
    df['above_median'] = (df['n_contributors'] >= df['n_contributors'].median()).astype(int)
    df['interaction'] = df['above_median'] * df['log_n']
    X2 = sm.add_constant(df[['log_n', 'above_median', 'interaction']])
    ols2 = sm.OLS(df['log_prs'], X2).fit()
    
    print(f"\n  OLS with size interaction:")
    print(f"    β (small teams): {ols2.params['log_n']:.4f}")
    print(f"    β shift (large teams): {ols2.params['interaction']:.4f}")
    print(f"    β (large teams): {ols2.params['log_n'] + ols2.params['interaction']:.4f}")
    print(f"    Interaction p-value: {ols2.pvalues['interaction']:.2e}")
    
    # Model 3: Mixed effects with random slope by size group
    try:
        # Random intercept + random slope by size group
        me = smf.mixedlm("log_prs ~ log_n", df, groups=df["size_group"],
                         re_formula="~log_n").fit(reml=True)
        
        print(f"\n  Mixed-Effects (random slope by size quintile):")
        print(f"    Fixed β = {me.fe_params['log_n']:.4f} ± {me.bse_fe['log_n']:.4f}")
        print(f"    Random slope variance: {me.cov_re.iloc[1,1]:.6f}")
        print(f"    Random slope SD: {np.sqrt(me.cov_re.iloc[1,1]):.4f}")
        print(f"    Log-likelihood: {me.llf:.1f}")
        print(f"    AIC: {me.aic:.1f}")
        
        # Extract group-specific slopes
        re = me.random_effects
        group_slopes = {}
        for g, effects in re.items():
            group_slope = me.fe_params['log_n'] + effects.iloc[1]
            n_range = df[df['size_group'] == g]['n_contributors']
            group_slopes[g] = {
                'beta': float(group_slope),
                'n_lo': int(n_range.min()),
                'n_hi': int(n_range.max()),
            }
            print(f"    Group {g} (N={n_range.min()}-{n_range.max()}): β = {group_slope:.3f}")
        
        me_result = {
            'fixed_beta': float(me.fe_params['log_n']),
            'fixed_se': float(me.bse_fe['log_n']),
            'random_slope_var': float(me.cov_re.iloc[1,1]),
            'aic': float(me.aic),
            'group_slopes': group_slopes,
        }
    except Exception as e:
        print(f"\n  Mixed-effects failed: {e}")
        print("  Falling back to stratified OLS...")
        me_result = {'error': str(e)}
        
        # Stratified OLS as fallback
        group_slopes = {}
        for g in sorted(df['size_group'].unique()):
            dg = df[df['size_group'] == g]
            if len(dg) < 20:
                continue
            s, i, r, p, se = linregress(dg['log_n'], dg['log_prs'])
            n_range = dg['n_contributors']
            group_slopes[int(g)] = {
                'beta': float(s), 'se': float(se),
                'n_lo': int(n_range.min()), 'n_hi': int(n_range.max()),
                'n_repos': len(dg),
            }
            print(f"    Quintile {g} (N={n_range.min()}-{n_range.max()}, "
                  f"n={len(dg)}): β = {s:.3f} ± {se:.3f}")
        me_result['group_slopes'] = group_slopes
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Analysis D: Scaling Exponent by Size Group',
                 fontsize=13, fontweight='bold')
    
    gs = me_result.get('group_slopes', {})
    if gs:
        groups = sorted(gs.keys())
        betas = [gs[g]['beta'] for g in groups]
        labels = [f"N={gs[g]['n_lo']}–{gs[g]['n_hi']}" for g in groups]
        colors = ['#2ecc71' if b < 1 else '#e74c3c' for b in betas]
        
        ax.bar(range(len(groups)), betas, color=colors, edgecolor='black')
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(1.0, color='red', ls='--', lw=1.5)
        ax.set_ylabel('β (scaling exponent)')
        ax.set_title('β by Size Quintile (Mixed-Effects or Stratified OLS)')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_mixed_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_mixed_effects.png")
    
    return {
        'ols_beta': float(ols.params.iloc[1]),
        'interaction_p': float(ols2.pvalues['interaction']),
        'beta_small': float(ols2.params['log_n']),
        'beta_large': float(ols2.params['log_n'] + ols2.params['interaction']),
        **me_result,
    }


# =========================================================================
# ANALYSIS E: Bootstrap Confidence Intervals
# =========================================================================
def analysis_e_bootstrap_ci(df):
    """Full bootstrap CI on the variance peak and its location."""
    print("\n" + "="*60)
    print("ANALYSIS E: Bootstrap Confidence Intervals on Variance Peak")
    print("="*60)
    
    edges = [5, 10, 20, 50, 100, 200, 500]
    n_boot = 2000
    
    peak_ratios = []
    peak_locations = []
    
    for b in range(n_boot):
        if (b + 1) % 500 == 0:
            print(f"  Bootstrap {b+1}/{n_boot}...")
        
        # Resample repos
        df_boot = df.sample(n=len(df), replace=True)
        
        variances = []
        for i in range(len(edges) - 1):
            mask = (df_boot['n_contributors'] >= edges[i]) & (df_boot['n_contributors'] < edges[i+1])
            df_bin = df_boot[mask]
            
            if len(df_bin) < 15:
                variances.append(np.nan)
                continue
            
            # Quick beta variance
            log_n = np.log(df_bin['n_contributors'].values)
            log_prs = np.log(df_bin['n_prs'].values.clip(min=1))
            
            betas = []
            for _ in range(200):
                idx = np.random.choice(len(log_n), len(log_n), replace=True)
                try:
                    s, _, _, _, _ = linregress(log_n[idx], log_prs[idx])
                    betas.append(s)
                except:
                    pass
            
            variances.append(np.var(betas) if len(betas) > 50 else np.nan)
        
        variances = np.array(variances)
        valid = ~np.isnan(variances)
        
        if valid.sum() >= 3 and variances[valid][0] > 0:
            peak_idx = np.nanargmax(variances)
            baseline = variances[valid][0]
            ratio = variances[peak_idx] / baseline
            midpoint = np.sqrt(edges[peak_idx] * edges[peak_idx + 1])
            
            peak_ratios.append(ratio)
            peak_locations.append(midpoint)
    
    peak_ratios = np.array(peak_ratios)
    peak_locations = np.array(peak_locations)
    
    ratio_ci = np.percentile(peak_ratios, [2.5, 50, 97.5])
    
    print(f"\n  Peak ratio: median = {ratio_ci[1]:.0f}×")
    print(f"  95% CI: [{ratio_ci[0]:.0f}×, {ratio_ci[2]:.0f}×]")
    
    # Where does the peak land?
    from collections import Counter
    loc_counts = Counter(peak_locations.astype(int))
    print(f"\n  Peak location distribution:")
    for loc, count in sorted(loc_counts.items()):
        pct = count / len(peak_locations) * 100
        print(f"    N ≈ {loc}: {pct:.1f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Analysis E: Bootstrap Confidence on Variance Peak',
                 fontsize=13, fontweight='bold')
    
    ax = axes[0]
    ax.hist(peak_ratios, bins=40, color='#8e44ad', edgecolor='black', alpha=0.7)
    ax.axvline(ratio_ci[1], color='red', lw=2, label=f'Median: {ratio_ci[1]:.0f}×')
    ax.axvline(ratio_ci[0], color='orange', ls='--', 
               label=f'95% CI: [{ratio_ci[0]:.0f}×, {ratio_ci[2]:.0f}×]')
    ax.axvline(ratio_ci[2], color='orange', ls='--')
    ax.set_xlabel('Peak variance ratio')
    ax.set_ylabel('Count')
    ax.set_title(f'Peak Ratio ({n_boot} bootstraps)')
    ax.legend()
    
    ax = axes[1]
    locs = sorted(loc_counts.keys())
    pcts = [loc_counts[l] / len(peak_locations) * 100 for l in locs]
    ax.bar(range(len(locs)), pcts, color='#3498db', edgecolor='black')
    ax.set_xticks(range(len(locs)))
    ax.set_xticklabels([f'N≈{l}' for l in locs])
    ax.set_ylabel('% of bootstraps')
    ax.set_title('Peak Location Stability')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_bootstrap_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_bootstrap_ci.png")
    
    return {
        'peak_ratio_median': float(ratio_ci[1]),
        'peak_ratio_ci': [float(ratio_ci[0]), float(ratio_ci[2])],
        'peak_location_distribution': {str(k): float(v/len(peak_locations)) 
                                        for k, v in loc_counts.items()},
        'n_bootstraps': n_boot,
    }


# =========================================================================
# ANALYSIS F: Confounder Proxy Check  
# =========================================================================
def analysis_f_confounders(df):
    """Check if variance peak survives after controlling for repo size proxies."""
    print("\n" + "="*60)
    print("ANALYSIS F: Confounder Proxy Check")
    print("="*60)
    
    # The main confounder we can check with available data:
    # comments/PR ratio as a proxy for code review intensity
    # This partly controls for repo "seriousness"
    
    df = df.copy()
    df['log_n'] = np.log(df['n_contributors'])
    df['log_prs'] = np.log(df['n_prs'].clip(lower=1))
    df['log_cpp'] = np.log(df['comments_per_pr'].clip(lower=0.01))
    
    edges = [5, 10, 20, 50, 100, 200, 500]
    labels = ['5-10', '10-20', '20-50', '50-100', '100-200', '200-500']
    
    # Residualize: regress log_prs on log_cpp first, then check if 
    # the variance pattern in residuals-vs-N persists
    from scipy.stats import linregress
    slope_cpp, int_cpp, _, _, _ = linregress(df['log_cpp'], df['log_prs'])
    df['log_prs_resid'] = df['log_prs'] - (slope_cpp * df['log_cpp'] + int_cpp)
    
    results_raw = []
    results_resid = []
    
    for i in range(len(edges) - 1):
        mask = (df['n_contributors'] >= edges[i]) & (df['n_contributors'] < edges[i+1])
        df_bin = df[mask]
        
        if len(df_bin) < 15:
            results_raw.append({'bin': labels[i], 'var': np.nan})
            results_resid.append({'bin': labels[i], 'var': np.nan})
            continue
        
        # Raw beta variance
        _, _, _, mean_raw = bootstrap_beta_variance(df_bin, n_boot=500)
        var_raw = bootstrap_beta_variance(df_bin, n_boot=500)[0]
        
        # Residualized beta variance (use residuals instead of raw log_prs)
        log_n = df_bin['log_n'].values
        log_prs_r = df_bin['log_prs_resid'].values
        
        betas_r = []
        for _ in range(500):
            idx = np.random.choice(len(log_n), len(log_n), replace=True)
            try:
                s, _, _, _, _ = linregress(log_n[idx], log_prs_r[idx])
                betas_r.append(s)
            except:
                pass
        var_resid = np.var(betas_r) if len(betas_r) > 50 else np.nan
        
        results_raw.append({'bin': labels[i], 'var': var_raw, 'n': len(df_bin)})
        results_resid.append({'bin': labels[i], 'var': var_resid, 'n': len(df_bin)})
    
    print("\n  Variance comparison (raw vs residualized):")
    print(f"  {'Bin':>12s} {'n':>6s} {'Var(raw)':>10s} {'Var(resid)':>10s}")
    print(f"  {'-'*42}")
    
    for raw, resid in zip(results_raw, results_resid):
        v_r = f"{raw['var']:.5f}" if raw.get('var') and not np.isnan(raw['var']) else "N/A"
        v_d = f"{resid['var']:.5f}" if resid.get('var') and not np.isnan(resid['var']) else "N/A"
        n = raw.get('n', '?')
        print(f"  {raw['bin']:>12s} {n:>6} {v_r:>10s} {v_d:>10s}")
    
    return {
        'method': 'residualized on log(comments_per_pr)',
        'raw_variances': {r['bin']: r['var'] for r in results_raw if not np.isnan(r.get('var', np.nan))},
        'resid_variances': {r['bin']: r['var'] for r in results_resid if not np.isnan(r.get('var', np.nan))},
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    np.random.seed(42)
    
    print("=" * 60)
    print("CS-047 CSD Robustness Suite")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    
    all_results = {}
    
    all_results['A_binning'] = analysis_a_binning_sensitivity(df)
    all_results['B_sliding'] = analysis_b_sliding_window(df)
    all_results['C_piecewise'] = analysis_c_piecewise(df)
    all_results['D_mixed'] = analysis_d_mixed_effects(df)
    all_results['E_bootstrap'] = analysis_e_bootstrap_ci(df)
    all_results['F_confounders'] = analysis_f_confounders(df)
    
    # Save all results
    with open(RESULTS_DIR / 'robustness_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    report_lines = [
        "CS-047 CSD Robustness Analysis — Summary",
        "=" * 50,
        "",
    ]
    
    # A: Binning
    report_lines.append("A. BINNING SENSITIVITY")
    for name, res in all_results['A_binning'].items():
        report_lines.append(f"  {name}: peak ratio = {res['peak_ratio']}× "
                          f"at {res['peak_bin']} (n={res['peak_n']})")
    report_lines.append("")
    
    # B: Sliding
    b = all_results['B_sliding']
    report_lines.append(f"B. SLIDING WINDOW: peak at N ≈ {b.get('peak_n', '?')}, "
                       f"ratio = {b.get('peak_ratio', '?')}×")
    report_lines.append(f"   β crosses 1.0 at N ≈ {b.get('crossing_n', '?')}")
    report_lines.append("")
    
    # C: Piecewise
    c = all_results['C_piecewise']
    report_lines.append(f"C. PIECEWISE: breakpoint N ≈ {c['breakpoint_n']:.0f} "
                       f"[{c['breakpoint_ci'][0]:.0f}, {c['breakpoint_ci'][1]:.0f}]")
    report_lines.append(f"   β_left = {c['beta_left']:.3f}, β_right = {c['beta_right']:.3f}")
    report_lines.append(f"   F = {c['f_stat']:.1f}, p = {c['p_value']:.2e}")
    report_lines.append("")
    
    # D: Mixed
    d = all_results['D_mixed']
    report_lines.append(f"D. MIXED-EFFECTS: interaction p = {d.get('interaction_p', '?')}")
    report_lines.append(f"   β (small teams) = {d.get('beta_small', '?')}")
    report_lines.append(f"   β (large teams) = {d.get('beta_large', '?')}")
    report_lines.append("")
    
    # E: Bootstrap
    e = all_results['E_bootstrap']
    report_lines.append(f"E. BOOTSTRAP: peak ratio = {e['peak_ratio_median']:.0f}× "
                       f"[{e['peak_ratio_ci'][0]:.0f}×, {e['peak_ratio_ci'][1]:.0f}×]")
    report_lines.append("")
    
    # Verdict
    report_lines.append("=" * 50)
    report_lines.append("VERDICT")
    report_lines.append("")
    
    # Check if peak is robust
    all_peaks = [v['peak_ratio'] for v in all_results['A_binning'].values() 
                 if v.get('peak_ratio') is not None]
    
    if all_peaks and min(all_peaks) > 5:
        report_lines.append("The variance peak is ROBUST across all binning strategies.")
        report_lines.append(f"Peak ratios range from {min(all_peaks):.0f}× to {max(all_peaks):.0f}×.")
    elif all_peaks and min(all_peaks) > 2:
        report_lines.append("The variance peak is PRESENT but ATTENUATED under some binnings.")
    else:
        report_lines.append("The variance peak is SENSITIVE to binning choices.")
    
    if c['p_value'] < 0.001:
        report_lines.append(f"Piecewise regression confirms breakpoint at N ≈ {c['breakpoint_n']:.0f} (p < 0.001).")
    
    report = "\n".join(report_lines)
    print(report)
    
    with open(RESULTS_DIR / 'robustness_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
