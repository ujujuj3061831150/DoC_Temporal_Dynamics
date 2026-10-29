"""
Temporal-MDCO / T-GECF: Order-Aware Temporal Dynamics & Consensus Fusion for Bedside EEG.
"""

from .operators import (
    compute_temporal_order_features,
    residual_bilateral_covariance,
    extract_clean_theta_series,
)
from .consensus import (
    SignedStandardizedMean,
    TemporalMDCO,
    AdaptiveGatedConsensus,
    TunedMLBaseline,
)
from .surrogates import (
    generate_synchronous_surrogates,
    verify_permutation_invariance,
)
from .adapters import (
    load_mendeley_temporal_dataset,
    compute_wielek_discrete_persistence,
)
from .figshare_adapter import (
    load_figshare_pdoc_dataset,
    get_figshare_group_partition,
)

__all__ = [
    "compute_temporal_order_features",
    "residual_bilateral_covariance",
    "extract_clean_theta_series",
    "SignedStandardizedMean",
    "TemporalMDCO",
    "AdaptiveGatedConsensus",
    "TunedMLBaseline",
    "generate_synchronous_surrogates",
    "verify_permutation_invariance",
    "load_mendeley_temporal_dataset",
    "compute_wielek_discrete_persistence",
    "load_figshare_pdoc_dataset",
    "get_figshare_group_partition",
]
