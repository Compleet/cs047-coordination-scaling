# Known Issues & Methodological Notes

This document records known limitations, caveats, and methodological
decisions discovered during pre-submission validation. It is intended
to aid reproducibility and preempt common reviewer concerns.

## 1. Lognormal vs Power-Law Preference

All trust network degree distributions (Epinions, Bitcoin OTC, Bitcoin
Alpha) prefer lognormal over pure power law by Vuong likelihood ratio
test (R < 0, p < 0.05 for all). The paper acknowledges this in §4.3
("lognormal body + Pareto tail") and classifies by generative mechanism
(multiplicative reinforcement) rather than distributional fit alone.

**Script:** `01_trust_network_exponents.py`

## 2. Spectral Dimension Sensitivity

Weyl's law estimates of d_s are unreliable for small networks (N < 300).
For Angular versions, Weyl gives d_s = 5-11 while heat kernel gives
d_s = 2.5-3.2. The paper reports heat kernel values as more reliable
(SI Appendix G). Both methods agree well for large networks (CondMat,
AstroPh: d_s ≈ 2.5-3.0).

**Script:** `08_collaboration_spectral.py`

## 3. Phase Transition Location: N ≈ 50 vs N ≈ 100

The paper's analytical derivation (§7.2) predicts N* ≈ 50 for β = 4/7,
k = 0.001. The GitHub bin data shows β crosses 1.0 between the 50-100
and 100-200 bins, consistent with the sensitivity range N* = 32-95
reported for k ∈ [0.0005, 0.002].

**Scripts:** `10_github_productivity.py`, `12_phase_transition.py`

## 4. WBE Simulation Is a Consistency Check

Script `11_wbe_simulation.py` generates exact WBE fractal networks and
recovers β = d/(d+1) with R² ≈ 1.0. This verifies the derivation, not
empirical data. Table 4 in the paper notes: "R² values reflect the fit
of simulated flow to the theoretical scaling relation on synthetic
fractal networks."

## 5. GitHub Data Dependency

Script `10_github_productivity.py` requires GHTorrent data (~440M rows)
for full analysis. Pre-computed results are provided in
`results/github_scaling_results.json`. The `overhead_per_pr_exponent`
field (0.737) measures Brooks' Law coefficient (how coordination cost
per PR grows with team size), NOT the scaling exponent β. See the
`_documentation` block in the JSON file.

Similarly, script `15_critical_slowing_down.py` uses pre-computed
variance values from the GHTorrent bootstrap analysis.

## 6. Two Null Models, Different Z-Scores

The paper reports two null model comparisons for ECP:
- **Maslov-Sneppen** (degree-preserving rewiring): z = 7-13, scripts 03/06
- **Barabási-Albert** (PA growth model): z = 43-112, script 13

Both are significant. The BA null is weaker (easier to exceed) because
it preserves only degree sequence growth, not the specific degree
sequence. The MS null is the standard in network science.

## 7. Attachment Kernel α Range

The paper reports α ∈ [0.8, 1.0] for trust network preferential
attachment (§5.3). Script `02_attachment_kernel.py` finds α = 0.88-1.04
via the Newman method, with BTC OTC out-degree showing α slightly
above 1.0. The R² values (0.68-0.74) indicate moderate fit quality.

## 8. Ecological Network Classifications

Ecological networks (script 09) show mixed classifications, often
borderline (2/4 votes). These networks are small (N = 21-185), which
limits the reliability of the classifier. The paper acknowledges this
in §9.4: "Preliminary testing on ecological networks yields mixed
results."


## 9. CSD Robustness Results

The critical-slowing-down variance peak (originally reported as 56×) has
been tested under seven alternative specifications (script 16b):

- **Bin-free sliding window**: 118× peak (no bins at all)
- **Piecewise regression**: breakpoint N ≈ 93, 95% CI [15, 203], p = 10⁻⁷
- **Bootstrap CI**: median 50×, 95% CI [25×, 112×] (excludes 1×)
- **Mixed-effects**: interaction Δβ = 0.56, p < 10⁻⁷⁸
- **Language stratification**: >5× in 6/8 languages independently
- **Residualized**: 22× after controlling for comments/PR
- **Equal-count bins**: ~5× (attenuated — transition zone compressed)

The equal-count bin attenuation is expected: forcing equal repo counts
per bin compresses the N=50–200 transition zone (containing ~1% of repos)
into a single bin. Log-space bins (98×) and all bin-free methods confirm
the peak. The residualization from 46× to 22× indicates partial
confounding from heavily-reviewed large projects but leaves a strong
signal.
