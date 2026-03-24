# Claude Code Prompt: CS-047 CSD Robustness Analysis
# ====================================================
# Copy everything below this line into Claude Code.

"""
I need you to run a two-phase CSD (critical slowing down) robustness
analysis on my local GHTorrent dataset for a PNAS submission.

## Context

The paper (CS-047) claims a 56× variance peak in scaling exponents at
team size N ≈ 100-200, interpreted as critical slowing down at a phase
transition. PNAS Statistical Review will challenge this on:
- bin sensitivity
- low-sample artifacts
- lack of continuous estimation
- no mixed-effects model
- no confounder controls

The scripts are already written and pushed to the repo.

## What to do

### Step 1: Clone/pull the repo
```bash
cd ~/projects  # or wherever you work
git clone https://github.com/Compleet/cs047-coordination-scaling.git
# or if already cloned:
cd cs047-coordination-scaling && git pull origin main
```

### Step 2: Install dependencies
```bash
pip install numpy scipy pandas statsmodels matplotlib scikit-learn
```

### Step 3: Phase 1 — Extract per-repo data (~2-3 hours)

Run the extraction script, pointing it at the directory containing
the GHTorrent CSV files:

```bash
python analysis/16a_extract_repos.py /path/to/ghtorrent/csvs/
```

This streams all CSVs (expects files named ghtorrent-*.csv or *.csv)
and produces `results/csd_robustness/repo_summary.csv` with columns:
repo, n_contributors, n_prs, n_comments, comments_per_pr, prs_per_person,
log_n, log_prs

It also saves extraction_metadata.json with row counts and filters.

Expected output: ~16,000-20,000 qualified repos (N≥5, comments≥20, PRs≥5).

If the CSV column names don't match (repo, actor_login, author_login,
pr_id), check the actual headers and adjust the script accordingly.

### Step 4: Phase 2 — Run robustness suite (~10-20 minutes)

```bash
python analysis/16b_csd_robustness.py
```

This runs six analyses (A through F) and produces:
- `results/csd_robustness/robustness_results.json` (all numbers)
- `results/csd_robustness/robustness_report.txt` (human summary)
- 5 PNG figures in `results/csd_robustness/`

### Step 5: Review and commit results

Read the robustness_report.txt. The key questions:

1. Does the variance peak survive equal-count bins? (Analysis A)
   - If the peak ratio drops below ~10× under equal-count binning,
     the CSD claim weakens significantly.

2. Does the sliding window show a continuous peak? (Analysis B)
   - If the peak is sharp and localized, good.
   - If it's a broad plateau, reframe as "transition zone" not "critical point."

3. Where does the piecewise breakpoint land? (Analysis C)
   - If the 95% CI includes 50, the N≈50 claim survives.
   - If it's firmly at 100-200, update the paper to say N≈100-200.

4. Does the mixed-effects interaction term reach significance? (Analysis D)
   - If interaction p < 0.01, the size-dependent β is real, not an artifact
     of heterogeneity.

5. Is the bootstrap CI tight? (Analysis E)
   - If the 95% CI on peak ratio includes 1×, the peak isn't real.
   - If it's [20×, 100×], the peak is robust.

6. Does residualizing on comments/PR change the picture? (Analysis F)
   - If the peak persists after controlling for review intensity,
     it's not driven by a few heavily-reviewed large projects.

### Step 6: Commit results

```bash
cd cs047-coordination-scaling
git add results/csd_robustness/
git commit -m "feat: CSD robustness results from GHTorrent full analysis

Phase 1: Extracted N repos from M total rows
Phase 2: Six robustness analyses completed

Key findings:
[FILL IN from robustness_report.txt]"
git push origin main
```

## Important notes

- The Phase 1 extraction is the bottleneck (~2-3h on 90GB).
  Phase 2 is fast (~10-20 min).
- If the CSV format is different from expected, the column mapping
  in 16a_extract_repos.py (lines ~50-65) needs adjustment.
  Look for columns like: repo_name, user_login, pull_request_id, etc.
- statsmodels is required for Analysis D (mixed-effects).
  If it fails, the script falls back to stratified OLS.
- All figures save as PNG at 150 dpi. Adjust in the script if needed.
- Random seed is fixed (42) for reproducibility.

## What happens next

Once the results are committed, I'll integrate the findings into the
paper (v8.1 → v8.2):
- Update Table 8 CSD status based on robustness results
- Add "Robustness" paragraph to §7.3 citing the six analyses
- Update SI Appendix L with the full robustness suite
- If breakpoint CI differs from N≈50, update §7.2 accordingly
"""
