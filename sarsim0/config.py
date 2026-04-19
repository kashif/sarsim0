"""
SarSim0 Configuration - Hyperparameters from Table 9 of the paper.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SarSimConfig:
    """Configuration for SarSim0 pipeline hyperparameters."""

    # SARIMA orders (Section 4.1)
    p_range: Tuple[int, int] = (0, 10)  # AR order
    q_range: Tuple[int, int] = (0, 3)  # MA order
    P_range: Tuple[int, int] = (0, 2)  # Seasonal AR order
    Q_range: Tuple[int, int] = (0, 2)  # Seasonal MA order
    s_range: Tuple[int, int] = (0, 52)  # Seasonal period

    # Seasonality pairs from Table 9: [[24,7], [7,52], [0,7], [0,4], [0,24], [0,52]]
    seasonality_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(24, 7), (7, 52), (0, 7), (0, 4), (0, 24), (0, 52)]
    )

    # Pole constraints for stability
    r_max: float = 0.9  # Non-seasonal AR pole radius max
    R_max: float = 0.1  # Seasonal AR pole radius max

    # Integration orders
    d_range: Tuple[float, float] = (0.0, 1.0)  # Fractional differencing
    D: int = 1  # Seasonal integration order (fixed)

    # Noiser parameters (Section 4.3, Table 9)
    # Base rates (λ₀) - LogUniform ranges
    poisson_lambda_range: Tuple[float, float] = (0.1, 100.0)
    gamma_lambda_range: Tuple[float, float] = (0.1, 100.0)
    lognormal_lambda_range: Tuple[float, float] = (0.1, 5.0)

    # Shape parameters (κ)
    gamma_kappa_range: Tuple[float, float] = (1.0, 50.0)  # LogUniform
    lognormal_kappa_range: Tuple[float, float] = (1.0, 3.0)  # LogUniform

    # Power for Generalized Gamma (ζ)
    gamma_zeta_range: Tuple[float, float] = (0.5, 1.5)  # Uniform

    # Generation parameters (Appendix E / Table 9)
    burn_in: int = 200
    series_length: int = 6000  # Total time series length (base sequence length, Table 9)

    # Window sizes — fully configurable; defaults match Table 9 of 2601.00970.
    # For Chronos-2 fine-tuning the pipeline default inference context_length is 2048,
    # but the SarSim0 paper trains with context_window=4096 and prediction_window=512.
    # Override these when fine-tuning for a specific horizon or context budget.
    context_window: int = 4096  # History sequence length (Table 9 default: 4096)
    prediction_window: int = 512  # Forecasting horizon   (Table 9 default: 512)
    vectorization_batch_size: int = 256  # For efficient on-the-fly generation

    # Padding augmentation (Table 9): zero-pad the left of the context window to simulate
    # shorter-history series.  pad_len ~ Uniform[0, min(pad_max, context_window-1)].
    # Paper value: Uniform[0, 4088]  (= context_window - 8, guaranteeing ≥8 real values).
    # Automatically clamped to context_window - 1 when context_window < pad_max + 1.
    # Set to 0 to disable padding.
    pad_max: int = 4088

    # Probability of multiplicative vs additive composition in SARIMA-2
    multiplicative_prob: float = 0.5

    # Input selection probability (SARIMA vs. SARIMA-2)
    sarima2_prob: float = 0.5

    # Multivariate generation (Phase 1)
    multivariate_prob: float = 0.3  # Probability of multivariate vs univariate
    n_variates_range: Tuple[int, int] = (2, 5)  # Number of variates when multivariate
    correlation_strength_range: Tuple[float, float] = (
        0.3,
        0.9,
    )  # Off-diagonal correlation

    # Covariate generation (Phase 2)
    with_covariates_prob: float = 0.3  # Probability of including covariates
    n_past_covariates_range: Tuple[int, int] = (0, 3)  # Past-only covariates
    n_future_covariates_range: Tuple[int, int] = (0, 2)  # Known-future covariates
