#!/usr/bin/env python3
"""
Regression test for script 08: spectral dimension computation.

Verifies that:
1. Normalized Laplacian is used (not unnormalized)
2. Angular heat-kernel d_s values match paper claims (2.93 +/- 0.28)
3. Implied beta values are in Class T range (< 1)

Run: python3 analysis/test_08_spectral.py
"""

import sys
import numpy as np
import importlib.util

# Import script 08
spec = importlib.util.spec_from_file_location(
    's08', 'analysis/08_collaboration_spectral.py')
s08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s08)


def test_uses_normalized_laplacian():
    """Verify source code uses normalized_laplacian_matrix, not laplacian_matrix."""
    with open('analysis/08_collaboration_spectral.py') as f:
        source = f.read()
    
    # Should NOT contain bare laplacian_matrix (except in comments/docstrings)
    import re
    # Find all nx.laplacian_matrix calls outside of comments
    bare_calls = re.findall(r'^\s+L\s*=\s*nx\.laplacian_matrix', source, re.MULTILINE)
    norm_calls = re.findall(r'nx\.normalized_laplacian_matrix', source)
    
    assert len(bare_calls) == 0, \
        f"Found {len(bare_calls)} uses of unnormalized nx.laplacian_matrix(). " \
        f"Paper Materials & Methods specifies normalized Laplacian [24]."
    assert len(norm_calls) >= 2, \
        f"Expected >=2 uses of nx.normalized_laplacian_matrix(), found {len(norm_calls)}"
    
    print("  [PASS] Uses normalized Laplacian")


def test_angular_ds_matches_paper():
    """Angular d_s(HK) should be ~2.93 +/- 0.28 (paper SI Appendix G)."""
    PAPER_DS_MEAN = 2.93
    PAPER_DS_STD = 0.28
    TOLERANCE = 0.5  # generous tolerance for sampling/numerical variation
    
    versions = ['v2', 'v5', 'v8', 'v11']
    ds_values = []
    
    for v in versions:
        try:
            G = s08.load_angular_version(v)
            d_s_hk, r2 = s08.compute_spectral_dimension_heat_kernel(G)
            ds_values.append(d_s_hk)
        except Exception as e:
            print(f"  [WARN] Could not load Angular {v}: {e}")
    
    assert len(ds_values) >= 3, \
        f"Need at least 3 Angular versions, got {len(ds_values)}"
    
    mean_ds = np.mean(ds_values)
    assert abs(mean_ds - PAPER_DS_MEAN) < TOLERANCE, \
        f"Mean d_s(HK) = {mean_ds:.2f}, expected {PAPER_DS_MEAN} +/- {TOLERANCE}. " \
        f"If using unnormalized Laplacian, d_s ~ 0.8 (wrong)."
    
    print(f"  [PASS] Angular d_s(HK) = {mean_ds:.2f} (paper: {PAPER_DS_MEAN} +/- {PAPER_DS_STD})")


def test_angular_beta_class_t():
    """All Angular implied beta values should be < 1 (Class T)."""
    versions = ['v2', 'v5', 'v8', 'v11']
    
    for v in versions:
        try:
            G = s08.load_angular_version(v)
            d_s_hk, _ = s08.compute_spectral_dimension_heat_kernel(G)
            beta = d_s_hk / (d_s_hk + 1)
            assert beta < 1.0, \
                f"Angular {v}: beta = {beta:.3f} >= 1.0 (not Class T)"
            assert beta > 0.5, \
                f"Angular {v}: beta = {beta:.3f} < 0.5 (too low, check Laplacian)"
        except FileNotFoundError:
            pass  # Skip missing versions
    
    print("  [PASS] All Angular beta values in Class T range (0.5 < beta < 1.0)")


def test_unnormalized_gives_wrong_answer():
    """Sanity check: unnormalized Laplacian should give d_s ~ 0.8 (wrong)."""
    import networkx as nx
    from scipy.linalg import eigh
    
    try:
        G = s08.load_angular_version('v2')
    except FileNotFoundError:
        print("  [SKIP] Angular v2 not available")
        return
    
    # Compute with unnormalized (the old bug)
    L_unnorm = nx.laplacian_matrix(G).astype(float).toarray()
    evals = np.sort(np.real(eigh(L_unnorm, eigvals_only=True)))
    evals_pos = evals[evals > 1e-10]
    lam_max, lam_min = evals_pos[-1], evals_pos[0]
    t_range = np.logspace(np.log10(0.1/lam_max), np.log10(10.0/lam_min), 50)
    P_t = np.array([np.sum(np.exp(-evals_pos * t)) for t in t_range])
    valid = (P_t > 0) & np.isfinite(P_t)
    lt, lp = np.log(t_range[valid]), np.log(P_t[valid])
    s, e = int(0.2*len(lt)), int(0.8*len(lt))
    slope, _ = np.polyfit(lt[s:e], lp[s:e], 1)
    ds_unnorm = -2 * slope
    
    assert ds_unnorm < 1.5, \
        f"Unnormalized Laplacian gave d_s = {ds_unnorm:.2f}, expected < 1.5"
    
    # Now verify the fixed function gives the right answer
    ds_norm, _ = s08.compute_spectral_dimension_heat_kernel(G)
    assert ds_norm > 2.0, \
        f"Normalized Laplacian gave d_s = {ds_norm:.2f}, expected > 2.0"
    
    print(f"  [PASS] Unnormalized d_s = {ds_unnorm:.2f} (wrong), "
          f"Normalized d_s = {ds_norm:.2f} (correct)")


if __name__ == '__main__':
    print("=" * 60)
    print("Regression tests for 08_collaboration_spectral.py")
    print("=" * 60)
    
    tests = [
        test_uses_normalized_laplacian,
        test_angular_ds_matches_paper,
        test_angular_beta_class_t,
        test_unnormalized_gives_wrong_answer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
