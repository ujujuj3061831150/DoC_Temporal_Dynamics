# -*- coding: utf-8 -*-
with open("scripts/build_temporal_dynamics_doi_package.py", "r", encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")

old_sec5_block = """    # 5.1 Mendeley Metrics Summary
    metrics_csv = ROOT / "results/temporal_dynamics_v1/metrics_summary.csv"
    if metrics_csv.exists():
        import pandas as pd
        df_m = pd.read_csv(metrics_csv)
        def get_auc(m_name, f_name):
            sub = df_m[(df_m["model"] == m_name) & (df_m["feature_set"] == f_name)]
            return float(sub["auroc"].values[0]) if not sub.empty else None

        a_can = get_auc("signed_standardized_mean", "canonical")
        a_lag = get_auc("signed_standardized_mean", "plus_lag1")
        a_res = get_auc("signed_standardized_mean", "residual_corr")
        
        assert a_can is not None and np.isclose(a_can, 0.8200, atol=1e-3)
        print(f"  Cell A: Static Baseline              {a_can:.4f}    0.8200    Verified (Dynamic)")
        
        assert a_lag is not None and np.isclose(a_lag, 0.8911, atol=1e-3)
        print(f"  Cell C: Temporal Linear (rho_1)      {a_lag:.4f}    0.8911    Verified (Dynamic)")"""

new_sec5_block = """    # 5.1 Mendeley Metrics Summary
    metrics_csv = ROOT / "results/temporal_dynamics_v1/metrics_summary.csv"
    if metrics_csv.exists():
        import pandas as pd
        df_m = pd.read_csv(metrics_csv)
        def get_auc(m_key):
            sub = df_m[df_m["model"] == m_key]
            return float(sub["macro_auroc"].values[0]) if not sub.empty else None

        a_can = get_auc("canonical__C")
        a_lag = get_auc("plus_lag1__C")
        a_comb = get_auc("plus_lag1_residual__C")
        a_mdco = get_auc("plus_lag1_residual__D")
        
        assert a_can is not None and np.isclose(a_can, 0.8200, atol=1e-3)
        print(f"  Cell A: Static Baseline              {a_can:.4f}    0.8200    Verified (Dynamic)")
        
        assert a_lag is not None and np.isclose(a_lag, 0.8911, atol=1e-3)
        print(f"  Cell C: Temporal Linear (rho_1)      {a_lag:.4f}    0.8911    Verified (Dynamic)")

        assert a_comb is not None and np.isclose(a_comb, 0.8996, atol=1e-3)
        print(f"  Combined Model (Unconstrained)       {a_comb:.4f}    0.8996    Verified (Dynamic)")

        assert a_mdco is not None and np.isclose(a_mdco, 0.8949, atol=1e-3)
        print(f"  Combined Model (MDCO)                {a_mdco:.4f}    0.8949    Verified (Dynamic)")"""

assert old_sec5_block in text, "old_sec5_block not found"
text = text.replace(old_sec5_block, new_sec5_block)

with open("scripts/build_temporal_dynamics_doi_package.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated build_temporal_dynamics_doi_package.py with exact metrics columns")
