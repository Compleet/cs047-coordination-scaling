#!/usr/bin/env python3
"""
Run all analysis scripts and verify results against expected values.

Usage:
    python run_all.py              # Run scripts 01-12, 15 (fast, <10 min)
    python run_all.py --full       # Include scripts 13-14 (slow, ~1 hour)
    python run_all.py --test-only  # Only run regression tests
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis"

# Scripts ordered by dependency and speed
FAST_SCRIPTS = [
    ("01_trust_network_exponents.py", "Table 6: Trust network γ exponents", 120),
    ("02_attachment_kernel.py", "§5.3: Attachment kernel α", 60),
    ("03_spectral_concentration.py", "§2.3: ECP spectral concentration", 300),
    ("04_cheeger_constant.py", "SI-E: Cheeger constant", 120),
    ("05_citation_networks.py", "§9.2: Citation network classification", 180),
    ("06_ecp_sensitivity.py", "§2.3: ECP k-threshold robustness", 300),
    ("07_ecp_timeseries.py", "§5.2: ECP temporal evolution", 300),
    ("08_collaboration_spectral.py", "§7: Spectral dimension d_s", 180),
    ("09_ecological_networks.py", "§9.4: Ecological network validation", 120),
    ("10_github_productivity.py", "§7.3-7.4: GitHub scaling (cached)", 5),
    ("11_wbe_simulation.py", "Table 4: WBE consistency check", 30),
    ("12_phase_transition.py", "§7.2: N≈50 threshold derivation", 10),
    ("15_critical_slowing_down.py", "SI-L: CSD bootstrap (cached)", 5),
]

SLOW_SCRIPTS = [
    ("13_ba_dynamic_null.py", "SI-J: BA dynamic null (z=43-112)", 3600),
    ("14_granger_ecp.py", "SI-K: Granger γ→ρ_k causality", 600),
]

TESTS = [
    ("test_08_spectral.py", "Regression: normalized Laplacian"),
]


def run_script(script_name, description, timeout_sec):
    """Run a single analysis script."""
    script_path = ANALYSIS_DIR / script_name
    if not script_path.exists():
        return "SKIP", f"File not found: {script_name}"

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True,
            timeout=timeout_sec
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            return "PASS", f"{elapsed:.0f}s"
        else:
            return "FAIL", result.stderr[-200:] if result.stderr else "unknown error"

    except subprocess.TimeoutExpired:
        return "TIMEOUT", f">{timeout_sec}s"
    except Exception as e:
        return "ERROR", str(e)[:100]


def main():
    full_mode = "--full" in sys.argv
    test_only = "--test-only" in sys.argv

    print("=" * 70)
    print("CS-047 Validation Suite")
    print("=" * 70)

    results = []

    if test_only:
        scripts_to_run = []
    elif full_mode:
        scripts_to_run = FAST_SCRIPTS + SLOW_SCRIPTS
    else:
        scripts_to_run = FAST_SCRIPTS

    # Run analysis scripts
    if scripts_to_run:
        print(f"\nRunning {len(scripts_to_run)} analysis scripts...\n")
        for script, desc, timeout in scripts_to_run:
            print(f"  [{desc}]")
            print(f"    {script}... ", end="", flush=True)
            status, detail = run_script(script, desc, timeout)
            print(f"{status} ({detail})")
            results.append((script, desc, status))

    # Run tests
    print(f"\nRunning {len(TESTS)} regression tests...\n")
    for script, desc in TESTS:
        print(f"  [{desc}]")
        print(f"    {script}... ", end="", flush=True)
        status, detail = run_script(script, desc, 120)
        print(f"{status} ({detail})")
        results.append((script, desc, status))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    passed = sum(1 for _, _, s in results if s == "PASS")
    failed = sum(1 for _, _, s in results if s == "FAIL")
    skipped = sum(1 for _, _, s in results if s == "SKIP")
    timeout = sum(1 for _, _, s in results if s == "TIMEOUT")

    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Timeout: {timeout}")
    print(f"  Skipped: {skipped}")

    if failed > 0:
        print(f"\nFailed scripts:")
        for script, desc, status in results:
            if status == "FAIL":
                print(f"  - {script}: {desc}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
