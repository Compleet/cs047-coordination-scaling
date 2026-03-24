# CS-047 GitHub CSD Robustness Analysis — Claude Code Workplan

## What This Is

The paper claims a 56× variance peak at N ≈ 100–200 in GitHub repo scaling
exponents, interpreted as critical slowing down at a phase transition. A PNAS
Statistical Review Committee will ask:

1. Is the peak robust to binning choices?
2. Is it a low-sample artifact (n=127 and n=60 in the peak bins)?
3. What does a continuous model show vs coarse bins?
4. Does the result hold after controlling for confounders?
5. What does a proper mixed-effects specification give?

This script answers all five.

## Prerequisites

- GHTorrent PR comments CSVs (~90GB) in a directory
- Python 3.10+ with: numpy, scipy, pandas, statsmodels, matplotlib
- pip install statsmodels scikit-learn

## Two-Phase Approach

### Phase 1: Extract per-repo summary (runs once, ~2-3 hours on 90GB)

Streams all CSVs and outputs a single `repo_summary.csv` with one row per repo:
  repo, n_contributors, n_prs, n_comments, comments_per_pr, prs_per_person

This is the reusable intermediate — all robustness analyses run on this file.

### Phase 2: Robustness analyses (runs on repo_summary.csv, ~10 min)

Six analyses, each producing figures and numerical results:

A. Alternative binning (equal-count quantile bins)
B. Sliding window variance estimation  
C. Piecewise regression (breakpoint detection)
D. Mixed-effects / hierarchical model
E. Bootstrap variance with confidence intervals
F. Confounder sensitivity (if metadata available)

## Output

All results go to `results/csd_robustness/` with:
- `repo_summary.csv` (Phase 1 extract)
- `robustness_report.txt` (human-readable summary)
- `fig_binning_sensitivity.png`
- `fig_sliding_window.png`
- `fig_piecewise_regression.png`
- `fig_mixed_effects.png`
- `fig_bootstrap_ci.png`
- `robustness_results.json` (machine-readable for paper integration)
