# sarsim0

`sarsim0` is a Python library for synthetic time-series generation based on the SarSim0 paper, "Zero-shot Forecasting by Simulation Alone", accepted at ICLR 2026.

The library implements the three-stage simulator described in the paper:

1. `S`: stable SARIMA trajectory generation with pole-based sampling
2. `I`: SARIMA-2 bi-seasonal composition for richer multi-seasonal traces
3. `N`: rate-based heavy-tailed noisers for burstiness and intermittency

It is designed for on-the-fly generation in PyTorch training loops and already includes tests for the current implementation.

The core univariate simulator and default hyperparameters follow the paper. The multivariate generation, Chronos-2 batch packing, and covariate-aware training path in this repository are implementation extensions built on top of the paper so the simulator can be used for newer universal forecasting workflows.

## Repository Scope

This repository implements the paper-faithful univariate SarSim0 core and extends it in a few practical directions for modern training pipelines.

Paper-faithful core:

- stable SARIMA sampling with paper-aligned defaults from Table 9 / Appendix E.3
- SARIMA-2 seasonal composition
- heavy-tailed noisers
- univariate on-the-fly training batches

Repository extensions:

- multivariate generation with correlated innovations
- mixed univariate / multivariate streaming batches for Chronos-2-style training
- covariate-aware generation with past-only and known-future covariates
- direct Chronos-2 batch packing with `context`, `future_target`, `future_covariates`, `group_ids`, and `num_output_patches`
- package structure, tests, and installable pip distribution

## Features

- Vectorized univariate SARIMA simulation
- SARIMA-2 additive and multiplicative seasonal composition
- Heavy-tailed noisers: Poisson, Gamma, Lognormal, passthrough
- Streaming `IterableDataset` and `DataLoader` helpers for training
- Multivariate generation with correlated innovations
- Mixed univariate / multivariate Chronos-2 style batch generation
- Optional correlated past and future covariate generation

## Install

From the repository:

```bash
pip install .
```

For development and tests:

```bash
pip install -e .[dev]
```

Main runtime dependencies:

- `torch`
- `numpy`
- `scipy`
- `numba`

## Quick Start

Generate paper-style `(context, target)` batches:

```python
from sarsim0 import SarSim0Generator, SarSimConfig

config = SarSimConfig(
    context_window=4096,
    prediction_window=512,
)

generator = SarSim0Generator(config=config, seed=42)
context, target = generator.generate_batch(batch_size=32)

print(context.shape)  # (32, 4096)
print(target.shape)   # (32, 512)
```

Create a PyTorch dataloader for on-the-fly generation:

```python
from sarsim0 import create_dataloader

loader = create_dataloader(
    batch_size=64,
    seed=42,
    num_workers=0,
    num_batches_per_epoch=10,
)

for context, target in loader:
    pass
```

Use the infinite streaming dataset for trainer-style dict batches:

```python
from sarsim0 import create_streaming_dataloader

loader = create_streaming_dataloader(
    context_length=512,
    prediction_length=64,
    batch_size=32,
    seed=42,
    num_workers=2,
)

batch = next(iter(loader))
print(batch.keys())
```

Enable the repository extension for multivariate plus covariate-aware Chronos-2 batches:

```python
from sarsim0 import SarSimConfig, create_streaming_dataloader

config = SarSimConfig(
    multivariate_prob=0.5,
    with_covariates_prob=0.5,
    n_variates_range=(2, 5),
    n_past_covariates_range=(0, 3),
    n_future_covariates_range=(0, 2),
)

loader = create_streaming_dataloader(
    context_length=512,
    prediction_length=64,
    batch_size=32,
    config=config,
    seed=42,
    num_workers=0,
)

batch = next(iter(loader))
print(batch["context"].shape)
print(batch["future_target"].shape)
print(batch["future_covariates"].shape)
print(batch["group_ids"].shape)
print(batch["num_output_patches"])
```

Generate raw series directly:

```python
from sarsim0 import SarSim0Generator

generator = SarSim0Generator(seed=42)
y = generator.generate_series(batch_size=8, length=1000)

print(y.shape)  # (8, 1000)
```

## Training with Chronos-2

The streaming dataloader emits the batch format expected by the Chronos-2 training path used in the original SarSim0 extension work: `context`, `future_target`, `future_covariates`, `group_ids`, and `num_output_patches`.

- when `with_covariates_prob=0`, `future_covariates` is an all-`NaN` placeholder and the fast vectorized path is used
- when `with_covariates_prob>0`, the repository uses the extended Chronos-2 packing path for mixed univariate, multivariate, and covariate-aware tasks
- `group_ids` assigns the same id to all rows belonging to the same multivariate task

Minimal trainer wiring:

```python
from chronos.chronos2.trainer import Chronos2Trainer
from sarsim0 import SarSimConfig, SarSim0StreamingDataset
from transformers import TrainingArguments

config = SarSimConfig(
    multivariate_prob=0.5,
    with_covariates_prob=0.5,
)

train_dataset = SarSim0StreamingDataset(
    context_length=512,
    prediction_length=64,
    batch_size=16,
    config=config,
    seed=42,
)

# Replace this with an actual Chronos-2 model.
model = ...

trainer = Chronos2Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=16,
        max_steps=1000,
        remove_unused_columns=False,
        report_to=[],
    ),
    train_dataset=train_dataset,
)

trainer.train()
```

This repository’s multivariate and covariate-aware Chronos-2 training support is an implementation extension. The published paper itself describes a univariate simulator core.

## Public API

Core configuration and generation:

- `SarSimConfig`
- `SarSim0Generator`
- `SarSim0Dataset`
- `SarSim0StreamingDataset`
- `create_dataloader`
- `create_streaming_dataloader`
- `generate_sarsim0_batch`
- `generate_sarima_batch`

Composition and noisers:

- `CompositionMode`
- `apply_sarima2_vectorized`
- `additive_compose`
- `multiplicative_compose`
- `NoiserType`
- `apply_noiser_vectorized`

Multivariate and covariates:

- `generate_multivariate_sarima_batch`
- `generate_multivariate_sarsim0_batch`
- `sample_correlation_matrix`
- `generate_with_covariates`
- `generate_mixed_with_covariates`
- `generate_sarsim0_chronos2_with_covariates`

## Defaults

The default configuration in `SarSimConfig` follows Table 9 / Appendix E.3 of the paper for the core simulator:

- `series_length=6000`
- `burn_in=200`
- `context_window=4096`
- `prediction_window=512`
- `seasonality_pairs=[(24, 7), (7, 52), (0, 7), (0, 4), (0, 24), (0, 52)]`

These can all be overridden when constructing `SarSimConfig`.

Repository-specific extensions:

- `multivariate_prob`, `n_variates_range`, and correlation controls are not part of the original paper’s univariate specification
- `with_covariates_prob`, `n_past_covariates_range`, and `n_future_covariates_range` are extensions for Chronos-2-style covariate-aware training
- `create_streaming_dataloader(...)` can emit Chronos-2 trainer batches with `context`, `future_target`, `future_covariates`, `group_ids`, and `num_output_patches`

## Running Tests

```bash
pytest -q
```

The current repository test suite covers the core simulator, the streaming trainer batches, and the repository extension paths for multivariate and covariate-aware Chronos-2-style training.

## Repository Layout

```text
sarsim0/
  config.py
  sarima.py
  sarima2.py
  noisers.py
  pipeline.py
  multivariate.py
  covariates.py
  streaming_dataset.py
tests/
pyproject.toml
```

## Status

This package is an implementation-oriented library based on the paper description and the current repository code. It is suitable for local research workflows, synthetic-data experiments, and model pretraining pipelines.

## Citation

If you use this repository, cite the original paper:

```bibtex
@inproceedings{oreshkin2026zeroshot,
    title={Zero-shot Forecasting by Simulation Alone},
    author={Boris N. Oreshkin and Mayank Jauhari and Ravi Kiran Selvam and Malcolm Wolff and Wenhao Pan and Shankar Ramasubramanian and Kin G. Olivares and Tatiana Konstantinova and Andres Potapczynski and Mengfei Cao and Dmitry Efimov and Michael W. Mahoney and Andrew Gordon Wilson},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=ZOLUTSU5gk}
}
```

And cite this code repository:

```bibtex
@software{Rasul_sarsim0,
    author = {Rasul, Kashif},
    license = {Apache-2.0},
    title = {{sarsim0}},
    url = {https://github.com/kashif/sarsim0}
}
```
