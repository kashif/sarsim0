"""
SARIMA-2 Bi-Seasonal Composition - Section 4.2 of the paper.

Combines high-frequency base signal with low-frequency envelope signal
using either additive or multiplicative composition.

Vectorized implementation using numpy/numba.
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from numba import njit, prange

from .config import SarSimConfig
from .sarima import generate_sarima_batch


class CompositionMode(Enum):
    """Composition mode for SARIMA-2."""

    ADDITIVE = 0
    MULTIPLICATIVE = 1


@njit(parallel=True, cache=True)
def normalize_to_range_batch(x: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    """Normalize each row to target range (vectorized)."""
    B, T = x.shape
    out = np.zeros((B, T), dtype=np.float64)

    for b in prange(B):  # ty: ignore[not-iterable]
        x_min = x[b, 0]
        x_max = x[b, 0]
        for t in range(1, T):
            if x[b, t] < x_min:
                x_min = x[b, t]
            if x[b, t] > x_max:
                x_max = x[b, t]

        x_range = x_max - x_min
        if x_range < 1e-10:
            x_range = 1.0

        for t in range(T):
            normalized = (x[b, t] - x_min) / x_range
            out[b, t] = normalized * (target_max - target_min) + target_min

    return out


@njit(parallel=True, cache=True)
def additive_compose_batch(y_base: np.ndarray, y_envelope: np.ndarray) -> np.ndarray:
    """Additive composition (Equation 7): y = y^(b) + y^(e)"""
    B, T = y_base.shape
    out = np.zeros((B, T), dtype=np.float64)

    for b in prange(B):  # ty: ignore[not-iterable]
        for t in range(T):
            out[b, t] = y_base[b, t] + y_envelope[b, t]

    return out


@njit(parallel=True, cache=True)
def multiplicative_compose_batch(
    y_base: np.ndarray,
    y_envelope_norm: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """
    Multiplicative composition (Equation 8):
    y = (1 + ω · ỹ^(e)) · y^(b)

    Where ỹ^(e) is envelope normalized to [-1, 1].
    """
    B, T = y_base.shape
    out = np.zeros((B, T), dtype=np.float64)

    for b in prange(B):  # ty: ignore[not-iterable]
        w = omega[b]
        for t in range(T):
            out[b, t] = (1.0 + w * y_envelope_norm[b, t]) * y_base[b, t]

    return out


def apply_sarima2_vectorized(
    y_base: torch.Tensor,
    config: SarSimConfig,
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
    envelope_s: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SARIMA-2 interaction to base signals (vectorized).

    For each series:
    1. Generate an envelope SARIMA signal
    2. Randomly choose additive or multiplicative composition
    3. Combine base and envelope

    Args:
        y_base: Base signals of shape (B, T)
        config: Configuration
        generator: PyTorch random generator
        device: Device for tensors
        envelope_s: Fixed seasonal period for the envelope. When called from
            generate_sarima2_paired_batch this is the second element of the
            sampled seasonality pair (paper Table 9). None → sample from s_range.

    Returns:
        Tuple of (composed signals, composition modes)
    """
    B, T = y_base.shape

    # Generate envelope signals; fixed_s pins the envelope to a specific seasonality.
    y_envelope, _ = generate_sarima_batch(B, T, config, generator, device, fixed_s=envelope_s)

    # Convert to numpy for fast processing
    y_base_np = y_base.cpu().numpy().astype(np.float64)
    y_envelope_np = y_envelope.cpu().numpy().astype(np.float64)

    # Randomly select composition mode for each series
    modes = torch.randint(0, 2, (B,), generator=generator, device=device)
    modes_np = modes.cpu().numpy()

    # Sample omega for multiplicative composition (use same seed as generator for reproducibility)
    omega_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
    omega_rng = np.random.default_rng(omega_seed)
    omega_np = omega_rng.uniform(0, 1, size=B)

    # Normalize envelope for multiplicative composition
    y_envelope_norm = normalize_to_range_batch(y_envelope_np, -1.0, 1.0)

    # Initialize output
    result_np = np.zeros((B, T), dtype=np.float64)

    # Separate indices by composition mode
    additive_mask = modes_np == CompositionMode.ADDITIVE.value
    mult_mask = modes_np == CompositionMode.MULTIPLICATIVE.value

    # Apply additive composition
    if additive_mask.any():
        add_result = additive_compose_batch(
            y_base_np[additive_mask],
            y_envelope_np[additive_mask],
        )
        result_np[additive_mask] = add_result

    # Apply multiplicative composition
    if mult_mask.any():
        mult_result = multiplicative_compose_batch(
            y_base_np[mult_mask],
            y_envelope_norm[mult_mask],
            omega_np[mult_mask],
        )
        result_np[mult_mask] = mult_result

    result = torch.from_numpy(result_np).float().to(device)
    return result, modes


def generate_sarima2_paired_batch(
    batch_size: int,
    length: int,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate SARIMA-2 series using paired seasonalities (paper-correct implementation).

    Samples one (s_base, s_env) pair from config.seasonality_pairs, generates the
    base signal with fixed seasonal period s_base and the envelope with s_env.
    This directly implements Table 9: "Seasonality pairs Uniform[[24,7],...]".

    Args:
        batch_size: Number of series
        length: Length of each series
        config: Configuration
        generator: PyTorch generator
        device: Output device

    Returns:
        Tensor of shape (batch_size, length)
    """
    if generator is None:
        generator = torch.Generator(device="cpu")

    # Sample one seasonality pair uniformly from the configured list.
    seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
    rng = np.random.default_rng(seed)
    pair_idx = int(rng.integers(0, len(config.seasonality_pairs)))
    s_base, s_env = config.seasonality_pairs[pair_idx]

    # Generate base with s_base, envelope with s_env.
    y_base, _ = generate_sarima_batch(batch_size, length, config, generator, device, fixed_s=s_base)
    y_composed, _ = apply_sarima2_vectorized(y_base, config, generator, device, envelope_s=s_env)

    return y_composed


def generate_sarima2_batch(
    batch_size: int,
    length: int,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, None]:
    """
    Generate SARIMA-2 series directly (base + envelope composition).

    Delegates to generate_sarima2_paired_batch so both base and envelope use the
    correct seasonality pairs from config (paper Table 9).

    Args:
        batch_size: Number of series
        length: Length of each series
        config: Configuration
        generator: PyTorch generator
        device: Output device

    Returns:
        Tuple of (tensor (batch_size, length), None)
    """
    return generate_sarima2_paired_batch(batch_size, length, config, generator, device), None


# Keep simple torch versions for compatibility
def normalize_to_range(
    x: torch.Tensor,
    target_min: float = -1.0,
    target_max: float = 1.0,
) -> torch.Tensor:
    """Normalize tensor to target range."""
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    x_range = x_max - x_min
    x_range = torch.where(x_range < 1e-10, torch.ones_like(x_range), x_range)
    normalized = (x - x_min) / x_range
    return normalized * (target_max - target_min) + target_min


def additive_compose(y_base: torch.Tensor, y_envelope: torch.Tensor) -> torch.Tensor:
    """Additive composition."""
    return y_base + y_envelope


def multiplicative_compose(
    y_base: torch.Tensor,
    y_envelope: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Multiplicative composition."""
    y_envelope_normalized = normalize_to_range(y_envelope, -1.0, 1.0)
    if omega.dim() == 0:
        return (1.0 + omega * y_envelope_normalized) * y_base
    else:
        omega = omega.unsqueeze(-1)
        return (1.0 + omega * y_envelope_normalized) * y_base
