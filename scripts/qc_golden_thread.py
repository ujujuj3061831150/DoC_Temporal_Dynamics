# -*- coding: utf-8 -*-
import json
import os
import re
import pandas as pd
import numpy as np

def run_golden_thread_audit():
    print("=" * 60)
    print("DIMENSION 1: NUMERICAL & STATISTICAL CONSISTENCY (GOLDEN THREAD)")
    print("=" * 60)

    # 1. Main metrics summary
    metrics_path = 'v3/results/temporal_dynamics_v1/metrics_summary.csv'
    df_metrics = pd.read_csv(metrics_path)
    
    # 2. Check 4-feature orthogonal suite
    # Look for 4-feature linear in metrics or contrasts
    print("\n--- Key Model Verifications ---")
    
    # Check hyperparam grid results
    with open('v3/results/temporal_dynamics_v1/hyperparam_grid_results.json', 'r') as f:
        hp_grid = json.load(f)
    print("Hyperparam Grid Top Results:")
    for k in ['soft_c_results', 'sigmoid_grid']:
        if k in hp_grid:
            print(f"  {k} keys/count: {len(hp_grid[k])}")

    # Check soft_c_results
    if 'soft_c_results' in hp_grid:
        c15 = hp_grid['soft_c_results'].get('1.5', {})
        print(f"  c=1.5: AUROC={c15.get('auroc')}, AP={c15.get('ap')}, BA={c15.get('balanced_accuracy')}, Sens={c15.get('sensitivity')}, Spec={c15.get('specificity')}")

    # Check CRRC benchmark summary
    crrc_path = 'v3/results/crrc_validation/crrc_benchmark_summary.json'
    if os.path.exists(crrc_path):
        with open(crrc_path, 'r') as f:
            crrc_data = json.load(f)
        print("\nCRRC Benchmark Summary:")
        print(f"  Cohort size: {crrc_data.get('cohort_size')}")
        print(f"  Protocol 1 (Clean 6-ch, N=13): AUROC={crrc_data.get('clean_6ch_auroc')}, Static={crrc_data.get('clean_6ch_static_baseline')}, Gain={crrc_data.get('clean_6ch_net_gain')}")
        print(f"  Protocol 2 (Full, N=15): AUROC={crrc_data.get('full_15_auroc')}, Static={crrc_data.get('full_15_static_baseline')}, Gain={crrc_data.get('full_15_net_gain')}")
        if 'supervised_ml_baselines' in crrc_data:
            print(f"  Supervised ML transfer collapse: {crrc_data['supervised_ml_baselines']}")

    # Check Figshare results
    fig_path = 'v3/results/temporal_dynamics_v1/figshare_validation_results.json'
    if os.path.exists(fig_path):
        with open(fig_path, 'r') as f:
            fig_data = json.load(f)
        print("\nFigshare Benchmark Summary:")
        for k, v in fig_data.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")

    # Check Wielek results
    wielek_path = 'v3/results/temporal_dynamics_v1/wielek_persistence_results.json'
    if os.path.exists(wielek_path):
        with open(wielek_path, 'r') as f:
            wielek_data = json.load(f)
        print("\nWielek Benchmark Summary:")
        for k, v in wielek_data.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")

    # 3. Check manuscript text numbers
    with open('v3/manuscript_temporal/manuscript.tex', 'r', encoding='utf-8') as f:
        ms_text = f.read()

    print("\n--- Verifying Core Numbers in manuscript.tex ---")
    checks = [
        ("0.9324", "4-Feature Linear AUROC"),
        ("0.8998", "4-Feature Linear AP"),
        ("+0.1024", "Direct margin vs strongest baseline"),
        ("+0.0311, +0.1856", "Direct 95% CI vs Ridge+rho1"),
        ("0.5702", "EEGNet empirical baseline AUROC"),
        ("0.8101", "4-Feature Linear Balanced Acc"),
        ("+0.0174, +0.2186", "4-Feature 95% CI"),
        ("0.9149", "5-Feature Combined Linear (c=1.5) AUROC"),
        ("0.9116", "T-GECF AUROC"),
        ("0.8669", "T-GECF PR-AUC (AP)"),
        ("+0.0711", "Representation Net Gain"),
        ("-0.0320", "Consensus Gap"),
        ("0.7778", "CRRC Hospital Cohort (N=15) AUROC"),
        ("+0.1296", "CRRC Hospital Net Gain"),
        ("0.7143", "CRRC Clean 6-Ch AUROC"),
        ("0.4286--0.4762", "Supervised ML Collapse on CRRC clean 6ch"),
        ("0.7691", "Figshare D1 AUROC"),
        ("77.59", "Figshare Balanced Accuracy"),
        ("+0.0492", "Figshare D1 net gain over static"),
        ("0.0302", "Wielek one-sided p-value"),
        ("0.0392", "Wielek nocturnal sleep p-value"),
        ("0.7348", "Wielek AUROC"),
        ("0.7576", "Wielek nocturnal AUROC"),
        ("0.42 \\pm 0.08", "Single-core CPU latency"),
        ("5{,}133", "Mendeley adjacent epoch pairs"),
        ("N=129", "Total multicenter records"),
        ("N=41", "Mendeley cohort size"),
        ("N=15", "CRRC cohort size"),
        ("N=50", "Figshare cohort size"),
        ("N=23", "Wielek cohort size"),
    ]

    all_passed = True
    for val, desc in checks:
        if val in ms_text:
            print(f"  [PASS] {desc}: '{val}' verified in text.")
        else:
            print(f"  [FAIL] {desc}: '{val}' NOT found in text!")
            all_passed = False

    print(f"\nAll Core Text Numbers Passed: {all_passed}")

if __name__ == '__main__':
    run_golden_thread_audit()
