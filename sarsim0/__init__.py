"""
SarSim0 - SARIMA Simulator for Zero-Shot Forecasting

A PyTorch-based synthetic time series generator implementing the SarSim0 pipeline
as described in the paper, with repository extensions for multivariate generation
and Chronos-2-style covariate-aware training batches. Generates diverse, realistic
time series for training foundation models.

Pipeline: y = N ∘ I ∘ S(ε)
- S: SARIMA base signal generator with pole-based stability
- I: SARIMA-2 bi-seasonal interaction layer
- N: Heavy-tailed noiser (Poisson, Gamma, Lognormal)

Example usage:
    from sarsim0 import SarSimConfig, create_dataloader, SarSim0Generator
    from sarsim0 import SarSim0StreamingDataset, create_streaming_dataloader

    # Tuple (context, target) dataloader — univariate, paper-style windows
    config = SarSimConfig()
    dataloader = create_dataloader(batch_size=256, config=config, num_workers=4)

    for context, target in dataloader:
        # context: (batch_size, 4096)
        # target: (batch_size, 512)
        ...

    # Chronos-2 style dict batches — vectorized mixed uni / multivariate
    stream = create_streaming_dataloader(
        512, 64, batch_size=32, num_workers=0, seed=42,
    )
    for batch in stream:
        # batch["context"], batch["future_target"], batch["group_ids"], ...
        ...

    # Or use generator directly
    generator = SarSim0Generator(config=config, seed=42)
    context, target = generator.generate_batch(batch_size=256)
"""

from importlib.metadata import PackageNotFoundError, version

from .config import SarSimConfig
from .covariates import (
    generate_mixed_with_covariates,
    generate_sarsim0_chronos2_with_covariates,
    generate_with_covariates,
)
from .multivariate import (
    generate_multivariate_sarima_batch,
    generate_multivariate_sarsim0_batch,
    sample_correlation_matrix,
)
from .noisers import (
    NoiserType,
    apply_noiser_vectorized,
    gamma_noiser,
    lognormal_noiser,
    poisson_noiser,
)
from .pipeline import (
    SarSim0Dataset,
    SarSim0Generator,
    create_dataloader,
    generate_mixed_sarsim0_chronos2,
    generate_sarsim0_batch,
)
from .sarima import generate_sarima_batch
from .sarima2 import (
    CompositionMode,
    additive_compose,
    apply_sarima2_vectorized,
    multiplicative_compose,
)
from .streaming_dataset import (
    SarSim0StreamingDataset,
    build_mixed_covariate_batch,
    build_vectorized_mixed_batch,
    create_streaming_dataloader,
)

try:
    __version__ = version("sarsim0")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # Config
    "SarSimConfig",
    # SARIMA
    "generate_sarima_batch",
    # SARIMA-2
    "CompositionMode",
    "apply_sarima2_vectorized",
    "additive_compose",
    "multiplicative_compose",
    # Noisers
    "NoiserType",
    "apply_noiser_vectorized",
    "poisson_noiser",
    "gamma_noiser",
    "lognormal_noiser",
    # Pipeline
    "generate_sarsim0_batch",
    "SarSim0Dataset",
    "SarSim0StreamingDataset",
    "SarSim0Generator",
    "build_mixed_covariate_batch",
    "build_vectorized_mixed_batch",
    "create_dataloader",
    "create_streaming_dataloader",
    "generate_mixed_sarsim0_chronos2",
    # Multivariate
    "generate_multivariate_sarima_batch",
    "generate_multivariate_sarsim0_batch",
    "sample_correlation_matrix",
    # Covariates
    "generate_with_covariates",
    "generate_mixed_with_covariates",
    "generate_sarsim0_chronos2_with_covariates",
]
