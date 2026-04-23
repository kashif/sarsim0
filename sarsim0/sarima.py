"""
SARIMA Simulator - Section 4.1 of the paper.

Generates synthetic time series using Seasonal ARIMA models with
pole-based stability guarantees.

Key insight from paper: "vectorize across trajectories: for each sampled
parameter tuple, we draw B independent sets of initial conditions (and
innovations), and we unroll B paths in parallel."

Uses scipy.signal.lfilter for fast O(T) ARMA filtering with minimal per-sample overhead.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import signal
from scipy.ndimage import convolve1d

from .config import SarSimConfig


def sample_poles(n_poles: int, r_max: float, rng: np.random.Generator) -> np.ndarray:
    """Sample poles uniformly by area within a disk of radius ``r_max``."""
    if n_poles == 0:
        return np.array([], dtype=np.complex128)
    # Draw radius from the correct area-uniform distribution: r^2 ~ Uniform(0, r_max^2).
    r = np.sqrt(rng.uniform(0.0, r_max**2, size=n_poles))
    theta = rng.uniform(0, 2 * np.pi, size=n_poles)
    return r * np.exp(1j * theta)


def poles_to_coeffs(poles: np.ndarray) -> np.ndarray:
    """Convert poles to polynomial coefficients."""
    if len(poles) == 0:
        return np.array([1.0])
    # np.poly returns coefficients of (x - p1)(x - p2)... = x^n + c1*x^(n-1) + ...
    result = np.poly(poles)
    return result.real


def expand_seasonal_poly(coeffs: np.ndarray, s: int) -> np.ndarray:
    """
    Expand seasonal polynomial coefficients.

    If coeffs = [1, -Φ₁, -Φ₂] and s=12, returns
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Φ₁, 0, ..., -Φ₂]
    """
    if s <= 1 or len(coeffs) <= 1:
        return coeffs

    # Length of expanded polynomial: 1 + (len(coeffs)-1) * s
    n_terms = len(coeffs)
    expanded_len = 1 + (n_terms - 1) * s
    expanded = np.zeros(expanded_len, dtype=np.float64)

    for i, c in enumerate(coeffs):
        expanded[i * s] = c

    return expanded


def build_sarima_polynomials(
    ar_coeffs: np.ndarray,
    ma_coeffs: np.ndarray,
    sar_coeffs: np.ndarray,
    sma_coeffs: np.ndarray,
    s: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build full SARIMA AR and MA polynomials.

    AR_full(L) = AR(L) * SAR(L^s)
    MA_full(L) = MA(L) * SMA(L^s)

    Returns (ar_poly, ma_poly) for use with lfilter.
    lfilter expects: b[0]*y[n] = a[0]*x[n] + a[1]*x[n-1] + ...
                     So we pass (ma_poly, ar_poly)
    """
    # Build AR polynomial: 1 - φ₁L - φ₂L² - ...
    # ar_coeffs are already [φ₁, φ₂, ...] (positive = subtract in standard form)
    ar_poly = np.array([1.0], dtype=np.float64)
    if len(ar_coeffs) > 0:
        ar_poly = np.concatenate([[1.0], -ar_coeffs])

    # Build MA polynomial: 1 + θ₁L + θ₂L² + ...
    ma_poly = np.array([1.0], dtype=np.float64)
    if len(ma_coeffs) > 0:
        ma_poly = np.concatenate([[1.0], ma_coeffs])

    # Build seasonal AR: 1 - Φ₁L^s - Φ₂L^{2s} - ...
    sar_poly = np.array([1.0], dtype=np.float64)
    if len(sar_coeffs) > 0 and s > 0:
        sar_poly_compressed = np.concatenate([[1.0], -sar_coeffs])
        sar_poly = expand_seasonal_poly(sar_poly_compressed, s)

    # Build seasonal MA: 1 + Θ₁L^s + Θ₂L^{2s} + ...
    sma_poly = np.array([1.0], dtype=np.float64)
    if len(sma_coeffs) > 0 and s > 0:
        sma_poly_compressed = np.concatenate([[1.0], sma_coeffs])
        sma_poly = expand_seasonal_poly(sma_poly_compressed, s)

    # Convolve to get full polynomials
    ar_full = np.convolve(ar_poly, sar_poly)
    ma_full = np.convolve(ma_poly, sma_poly)

    return ar_full, ma_full


def fractional_diff_coeffs(d: float, max_lag: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Compute fractional differencing coefficients (Hosking 1981)."""
    coeffs = [1.0]
    b_k = 1.0
    for k in range(1, max_lag + 1):
        b_k = b_k * (k - 1 - d) / k
        if abs(b_k) < tol:
            break
        coeffs.append(b_k)
    return np.array(coeffs, dtype=np.float64)


def apply_arma_lfilter_batch(
    epsilon: np.ndarray,
    ar_poly: np.ndarray,
    ma_poly: np.ndarray,
) -> np.ndarray:
    """
    Apply ARMA filtering using scipy.signal.lfilter (vectorized across batch).

    AR(L) * y_t = MA(L) * ε_t

    Args:
        epsilon: White noise (B, T)
        ar_poly: AR polynomial [1, -φ₁, -φ₂, ...]
        ma_poly: MA polynomial [1, θ₁, θ₂, ...]

    Returns:
        Filtered output (B, T)
    """
    B, T = epsilon.shape

    # lfilter signature: lfilter(b, a, x) solves a*y = b*x
    # For ARMA: AR(L)*y = MA(L)*ε, so b=ma_poly, a=ar_poly
    # lfilter works along axis=-1 by default

    # Apply lfilter to each row
    y = signal.lfilter(ma_poly, ar_poly, epsilon, axis=-1)

    return y


def apply_fractional_integration_batch(
    y: np.ndarray,
    d: float,
) -> np.ndarray:
    """
    Apply fractional integration (1-L)^(-d) using convolution.

    Uses scipy.ndimage.convolve1d which is vectorized.
    """
    if d <= 0:
        return y

    frac_coeffs = fractional_diff_coeffs(-d)  # -d for integration
    # convolve1d with origin adjustment for causal filter
    # mode='constant' with cval=0 is like zero-padding
    y_integrated = convolve1d(
        y, frac_coeffs, axis=-1, mode="constant", origin=-(len(frac_coeffs) // 2)
    )

    return y_integrated


def apply_seasonal_integration_batch(y: np.ndarray, D: int, s: int) -> np.ndarray:
    """
    Apply seasonal integration (1-L^s)^(-D) using lfilter.

    (1 - L^s)^(-1) is an IIR filter with a=1 and b=[1, 0, 0, ..., -1] (s zeros then -1).
    """
    if D == 0 or s == 0:
        return y

    # Build IIR filter for (1 - L^s)^(-1)
    # y[n] - y[n-s] = x[n] => y[n] = x[n] + y[n-s]
    # lfilter: a[0]*y[n] = b[0]*x[n] - a[1]*y[n-1] - ... - a[s]*y[n-s]
    # For (1-L^s)^(-1): a = [1, 0, 0, ..., 0, -1] (s+1 terms), b = [1]
    a = np.zeros(s + 1, dtype=np.float64)
    a[0] = 1.0
    a[s] = -1.0
    b = np.array([1.0], dtype=np.float64)

    out = y.copy()
    for _ in range(D):
        out = signal.lfilter(b, a, out, axis=-1)

    return out


def generate_sarima_vectorized(
    batch_size: int,
    length: int,
    ar_coeffs: np.ndarray,
    ma_coeffs: np.ndarray,
    sar_coeffs: np.ndarray,
    sma_coeffs: np.ndarray,
    s: int,
    d: float,
    D: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate B SARIMA series with SHARED parameters using fast lfilter.

    Args:
        batch_size: B trajectories
        length: T time steps
        ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs: Filter coefficients
        s: Seasonal period
        d: Fractional differencing order
        D: Seasonal integration order
        rng: Random generator

    Returns:
        Array of shape (batch_size, length)
    """
    # Generate innovations
    epsilon = rng.standard_normal((batch_size, length)).astype(np.float64)

    # Build full SARIMA polynomials
    ar_poly, ma_poly = build_sarima_polynomials(ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs, s)

    # Apply ARMA filter (fast lfilter)
    y = apply_arma_lfilter_batch(epsilon, ar_poly, ma_poly)

    # Fractional integration using scipy.signal.oaconvolve (fast for long sequences)
    if d > 0:
        # Clip BEFORE oaconvolve to prevent FFT overflow that corrupts scipy state
        y = np.clip(y, -1e6, 1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        frac_coeffs = fractional_diff_coeffs(-d)  # -d for integration
        # Use overlap-add convolution which is O(N log N) and vectorized
        y_conv = signal.oaconvolve(y, frac_coeffs.reshape(1, -1), mode="full", axes=-1)
        y = y_conv[:, :length]

    # Seasonal integration
    if D > 0 and s > 0:
        y = apply_seasonal_integration_batch(y, D, s)

    # Handle numerical issues (also clips post-integration values)
    y = np.clip(y, -1e6, 1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    return y


def generate_sarima_batch(
    batch_size: int,
    length: int,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    fixed_s: Optional[int] = None,
) -> Tuple[torch.Tensor, None]:
    """
    Generate batch of SARIMA series.

    Uses paper's approach: sample ONE set of parameters, generate B trajectories
    with different innovations (vectorized across batch).

    Args:
        batch_size: Number of series
        length: Length of each series
        config: Configuration
        generator: PyTorch generator for seed
        device: Output device

    Returns:
        Tuple of (tensor (batch_size, length), None)
    """
    if generator is not None:
        seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
    else:
        seed = int(torch.randint(0, 2**31, (1,)).item())

    rng = np.random.default_rng(seed)
    vec_batch = min(batch_size, config.vectorization_batch_size)

    results = []
    remaining = batch_size

    while remaining > 0:
        current_batch = min(remaining, vec_batch)

        # Sample ONE set of SARIMA parameters for this batch
        p = rng.integers(config.p_range[0], config.p_range[1] + 1)
        q = rng.integers(config.q_range[0], config.q_range[1] + 1)
        P = rng.integers(config.P_range[0], config.P_range[1] + 1)
        Q = rng.integers(config.Q_range[0], config.Q_range[1] + 1)
        s = (
            fixed_s
            if fixed_s is not None
            else rng.integers(config.s_range[0], config.s_range[1] + 1)
        )
        d = rng.uniform(config.d_range[0], config.d_range[1])

        # Stability mixture (paper Section 4.1): the joint AR polynomial A(L) = ϕ(L)·Φ(L^s)
        # has a non-convex stability region, so we ALWAYS force either the non-seasonal AR
        # (p=0, ϕ=0) or the seasonal AR (P=0, Φ=0) to zero — 50/50.
        if rng.random() < 0.5:
            p = 0  # seasonal-AR-only: ϕ_i = 0 for all i
        else:
            P = 0  # AR-only: Φ_j = 0 for all j

        # Sample poles and get coefficients
        ar_poles = sample_poles(p, config.r_max, rng)
        ar_coeffs_full = poles_to_coeffs(ar_poles)
        # Extract AR coefficients: [1, -φ₁, -φ₂, ...] -> [φ₁, φ₂, ...] (flip sign)
        ar_coeffs = -ar_coeffs_full[1:] if len(ar_coeffs_full) > 1 else np.array([])

        ma_poles = sample_poles(q, config.r_max, rng)
        ma_coeffs_full = poles_to_coeffs(ma_poles)
        # MA coefficients: [1, -θ₁, -θ₂, ...] -> [θ₁, θ₂, ...] (flip sign, since poles_to_coeffs gives -(root))
        ma_coeffs = -ma_coeffs_full[1:] if len(ma_coeffs_full) > 1 else np.array([])

        sar_poles = sample_poles(P, config.R_max, rng)
        sar_coeffs_full = poles_to_coeffs(sar_poles)
        sar_coeffs = -sar_coeffs_full[1:] if len(sar_coeffs_full) > 1 else np.array([])

        sma_poles = sample_poles(Q, config.R_max, rng)
        sma_coeffs_full = poles_to_coeffs(sma_poles)
        sma_coeffs = -sma_coeffs_full[1:] if len(sma_coeffs_full) > 1 else np.array([])

        # Generate batch with shared parameters
        y = generate_sarima_vectorized(
            current_batch,
            length,
            ar_coeffs,
            ma_coeffs,
            sar_coeffs,
            sma_coeffs,
            s,
            d,
            config.D,
            rng,
        )

        results.append(y)
        remaining -= current_batch

    y_all = np.concatenate(results, axis=0)
    return torch.from_numpy(y_all).float().to(device), None
