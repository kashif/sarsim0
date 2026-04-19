"""
Vectorized streaming IterableDataset for SarSim0.

Uses batched ``generate_sarsim0_batch`` / ``generate_multivariate_sarsim0_batch``
(one call per univariate block and one per multivariate block) instead of a
per-sample Python loop. Yields full training batches as dicts compatible with
Chronos-2 style trainers (``context``, ``future_target``, ``future_covariates``,
``group_ids``, ``num_output_patches``).
"""

from __future__ import annotations

import time
from typing import Iterator, Mapping, Optional, Tuple, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import SarSimConfig
from .covariates import Chronos2RawInput, CovariateMap, generate_mixed_with_covariates
from .multivariate import generate_multivariate_sarsim0_batch
from .pipeline import generate_sarsim0_batch


class Chronos2Batch(TypedDict):
    context: torch.Tensor
    future_target: torch.Tensor
    future_covariates: torch.Tensor
    group_ids: torch.Tensor
    num_output_patches: int


def build_vectorized_mixed_batch(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    config: SarSimConfig,
    generator: torch.Generator,
    device: torch.device,
    *,
    output_patch_size: int = 16,
) -> Chronos2Batch:
    """
    Build one training batch from vectorized SarSim0 generation.

    Splits ``batch_size`` into univariate vs multivariate groups (Bernoulli with
    ``config.multivariate_prob``), runs at most one batched univariate call and
    one batched multivariate call, then packs rows with ``group_ids`` (shared id
    per multivariate task).

    Args:
        batch_size: Number of *tasks* (univariate tasks count as 1; one multivariate
            draw contributes ``n_variates`` rows).
        context_length: History length.
        prediction_length: Forecast horizon.
        config: Must have ``series_length`` such that simulated length after burn-in
            is at least ``context_length + prediction_length`` (defaults are built
            that way when using :class:`SarSim0StreamingDataset`).
        generator: PyTorch RNG (CPU).
        device: Tensor device for generation.
        output_patch_size: Patch size used only to set ``num_output_patches``.

    Returns:
        Dict with float tensors ``context``, ``future_target``, ``future_covariates``
        (all-NaN placeholder), long ``group_ids``,         and Python int ``num_output_patches``.
    """
    total_gen = config.series_length + config.burn_in
    flags = torch.rand(batch_size, generator=generator, device=torch.device("cpu"))
    n_mv = int((flags < config.multivariate_prob).sum().item())
    n_uv = batch_size - n_mv

    max_start = max(
        1,
        config.series_length - config.context_window - config.prediction_window + 1,
    )

    context_rows: list[torch.Tensor] = []
    future_rows: list[torch.Tensor] = []
    group_id_rows: list[torch.Tensor] = []
    group_counter = 0

    if n_uv > 0:
        y_uv = generate_sarsim0_batch(n_uv, total_gen, config, generator, device=device)
        for i in range(n_uv):
            s = int(
                torch.randint(
                    0,
                    max_start,
                    (1,),
                    generator=generator,
                    device=torch.device("cpu"),
                ).item()
            )
            ctx = y_uv[i, s : s + config.context_window]
            fut = y_uv[
                i,
                s + config.context_window : s + config.context_window + config.prediction_window,
            ]
            context_rows.append(ctx.unsqueeze(0))
            future_rows.append(fut.unsqueeze(0))
            group_id_rows.append(
                torch.tensor([group_counter], dtype=torch.long, device=torch.device("cpu"))
            )
            group_counter += 1

    if n_mv > 0:
        n_variates = int(
            torch.randint(
                config.n_variates_range[0],
                config.n_variates_range[1] + 1,
                (1,),
                generator=generator,
                device=torch.device("cpu"),
            ).item()
        )
        y_mv = generate_multivariate_sarsim0_batch(
            n_mv, n_variates, total_gen, config, generator, device=device
        )
        for i in range(n_mv):
            s = int(
                torch.randint(
                    0,
                    max_start,
                    (1,),
                    generator=generator,
                    device=torch.device("cpu"),
                ).item()
            )
            ctx = y_mv[i, :, s : s + config.context_window]
            fut = y_mv[
                i,
                :,
                s + config.context_window : s + config.context_window + config.prediction_window,
            ]
            context_rows.append(ctx)
            future_rows.append(fut)
            group_id_rows.append(
                torch.full(
                    (n_variates,), group_counter, dtype=torch.long, device=torch.device("cpu")
                )
            )
            group_counter += 1

    context = torch.cat(context_rows, dim=0).float()
    future_target = torch.cat(future_rows, dim=0).float()
    group_ids = torch.cat(group_id_rows, dim=0).long()
    future_covariates = torch.full_like(future_target, float("nan"))

    num_output_patches = (prediction_length + output_patch_size - 1) // output_patch_size

    return {
        "context": context,
        "future_target": future_target,
        "future_covariates": future_covariates,
        "group_ids": group_ids,
        "num_output_patches": num_output_patches,
    }


def _prepare_single_chronos2_task(
    sample: Chronos2RawInput,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Convert a raw SarSim0 sample into Chronos-2 task tensors."""
    target = torch.from_numpy(cast(np.ndarray, sample["target"])).float()
    if target.ndim == 1:
        target = target.unsqueeze(0)

    n_targets = target.shape[0]
    past_covariates = cast(CovariateMap, sample.get("past_covariates", {}))
    future_covariates = cast(CovariateMap, sample.get("future_covariates", {}))

    past_tensors: list[torch.Tensor] = [target]
    past_only_cov_names = sorted(set(past_covariates.keys()) - set(future_covariates.keys()))
    future_cov_names = sorted(future_covariates.keys())

    for name in past_only_cov_names:
        cov = past_covariates[name]
        if isinstance(cov, np.ndarray):
            cov = torch.from_numpy(cov).float()
        past_tensors.append(cov.unsqueeze(0))

    for name in future_cov_names:
        cov = past_covariates[name]
        if isinstance(cov, np.ndarray):
            cov = torch.from_numpy(cov).float()
        past_tensors.append(cov.unsqueeze(0))

    task_past = torch.cat(past_tensors, dim=0)

    future_tensors: list[torch.Tensor] = []
    for name in future_cov_names:
        cov = future_covariates[name]
        if isinstance(cov, np.ndarray):
            cov = torch.from_numpy(cov).float()
        future_tensors.append(cov.unsqueeze(0))

    if future_tensors:
        task_future = torch.cat(future_tensors, dim=0)
    else:
        task_future = torch.zeros((0, 0), dtype=torch.float32)

    n_covariates = len(past_only_cov_names) + len(future_cov_names)
    n_future_covariates = len(future_cov_names)
    return task_past, task_future, n_targets, n_covariates, n_future_covariates


def build_mixed_covariate_batch(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    config: SarSimConfig,
    generator: torch.Generator,
    *,
    output_patch_size: int = 16,
) -> Chronos2Batch:
    """
    Build one Chronos-2-compatible batch including multivariate tasks and covariates.

    This path mirrors the original Chronos-2 SarSim0 fine-tuning wrapper: inputs are
    first generated in raw Chronos-2 dict form and then packed into the stacked batch
    representation expected by the Chronos-2 trainer.
    """
    samples: list[Chronos2RawInput] = generate_mixed_with_covariates(
        batch_size=batch_size,
        context_length=context_length,
        prediction_length=prediction_length,
        config=config,
        generator=generator,
    )

    context_rows: list[torch.Tensor] = []
    future_target_rows: list[torch.Tensor] = []
    future_covariate_rows: list[torch.Tensor] = []
    group_id_rows: list[torch.Tensor] = []

    for group_id, sample in enumerate(samples):
        task_past, task_future, n_targets, n_covariates, n_future_covariates = (
            _prepare_single_chronos2_task(sample)
        )
        n_past_only_covariates = n_covariates - n_future_covariates
        full_length = task_past.shape[-1]

        slice_idx = int(
            torch.randint(
                prediction_length,
                full_length - prediction_length + 1,
                (1,),
                generator=generator,
                device=torch.device("cpu"),
            ).item()
        )

        if slice_idx >= context_length:
            context = task_past[:, slice_idx - context_length : slice_idx]
        else:
            context = task_past[:, :slice_idx]

        future_target = task_past[:, slice_idx : slice_idx + prediction_length].clone()
        future_target[n_targets:] = torch.nan

        if n_future_covariates > 0:
            if task_future.shape[-1] == prediction_length:
                future_covs = task_future
            else:
                future_covs = task_past[
                    -n_future_covariates:,
                    slice_idx : slice_idx + prediction_length,
                ]
        else:
            future_covs = torch.zeros((0, prediction_length), dtype=torch.float32)

        future_padding = torch.full(
            (n_targets + n_past_only_covariates, prediction_length),
            torch.nan,
            dtype=torch.float32,
        )
        future_covariates = torch.cat([future_padding, future_covs], dim=0)

        context_rows.append(context.float())
        future_target_rows.append(future_target.float())
        future_covariate_rows.append(future_covariates.float())
        group_id_rows.append(torch.full((context.shape[0],), group_id, dtype=torch.long))

    max_context = max(row.shape[-1] for row in context_rows)
    padded_context_rows = []
    for row in context_rows:
        if row.shape[-1] < max_context:
            padding = torch.full(
                (row.shape[0], max_context - row.shape[-1]),
                torch.nan,
                dtype=row.dtype,
            )
            row = torch.cat([padding, row], dim=-1)
        padded_context_rows.append(row)

    return {
        "context": torch.cat(padded_context_rows, dim=0),
        "future_target": torch.cat(future_target_rows, dim=0),
        "future_covariates": torch.cat(future_covariate_rows, dim=0),
        "group_ids": torch.cat(group_id_rows, dim=0),
        "num_output_patches": (prediction_length + output_patch_size - 1) // output_patch_size,
    }


class SarSim0StreamingDataset(IterableDataset):
    """
    Infinite iterable dataset yielding vectorized SarSim0 batches for forecasting trainers.

    Each step yields one dict (a full logical batch). Use ``DataLoader(..., batch_size=None)``.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        batch_size: int,
        *,
        output_patch_size: int = 16,
        multivariate_prob: Optional[float] = None,
        config: Optional[SarSimConfig] = None,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = 42,
        reseed_interval: Optional[int] = None,
        use_time_based_seed_when_unset: bool = False,
    ):
        """
        Args:
            context_length: Context / history length.
            prediction_length: Forecast horizon.
            batch_size: Number of univariate/multivariate *tasks* per yielded batch
                (same notion as ``SarSim0StreamingChronos2Dataset`` with covariates off).
            output_patch_size: For ``num_output_patches`` in the batch dict.
            multivariate_prob: Overrides ``config.multivariate_prob`` when not None.
            config: Optional full :class:`SarSimConfig`. When None, a default is built
                with ``series_length=context_length + prediction_length`` and
                ``burn_in=200`` so total generated length matches the usual pipeline.
            device: Device for SarSim0 tensor ops.
            seed: Base RNG seed; ``None`` with ``use_time_based_seed_when_unset=True``
                uses wall-clock jitter (similar to training scripts); bare ``None``
                uses only worker id offset (weak variety).
            reseed_interval: If set, re-``manual_seed`` the generator every this many
                yielded batches (after the first batch).
            use_time_based_seed_when_unset: If True and ``seed is None``, add
                ``time.time()``-based bits when constructing worker seeds.
        """
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.output_patch_size = output_patch_size
        self.device = device
        self.seed = seed
        self.reseed_interval = reseed_interval
        self.use_time_based_seed_when_unset = use_time_based_seed_when_unset

        total_len = context_length + prediction_length
        self.config = config or SarSimConfig(
            series_length=total_len,
            context_window=context_length,
            prediction_window=prediction_length,
            burn_in=200,
            multivariate_prob=0.3,
            n_variates_range=(2, 5),
            with_covariates_prob=0.0,
        )
        if multivariate_prob is not None:
            self.config.multivariate_prob = multivariate_prob

        self.config.context_window = context_length
        self.config.prediction_window = prediction_length
        if self.config.series_length < total_len:
            self.config.series_length = total_len

    def __iter__(self) -> Iterator[Chronos2Batch]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0

        if self.seed is None:
            if self.use_time_based_seed_when_unset:
                base = (int(time.time() * 1000) % (2**20)) + wid
            else:
                base = wid
        else:
            base = self.seed + wid

        generator = torch.Generator(device="cpu")
        initial = base
        generator.manual_seed(initial)

        batch_num = 0
        while True:
            if self.reseed_interval and batch_num > 0 and batch_num % self.reseed_interval == 0:
                generator.manual_seed(initial + batch_num)

            if self.config.with_covariates_prob > 0.0:
                yield build_mixed_covariate_batch(
                    self.batch_size,
                    self.context_length,
                    self.prediction_length,
                    self.config,
                    generator,
                    output_patch_size=self.output_patch_size,
                )
            else:
                yield build_vectorized_mixed_batch(
                    self.batch_size,
                    self.context_length,
                    self.prediction_length,
                    self.config,
                    generator,
                    self.device,
                    output_patch_size=self.output_patch_size,
                )
            batch_num += 1


def create_streaming_dataloader(
    context_length: int,
    prediction_length: int,
    batch_size: int,
    *,
    output_patch_size: int = 16,
    multivariate_prob: Optional[float] = None,
    config: Optional[SarSimConfig] = None,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = 42,
    num_workers: int = 0,
    reseed_interval: Optional[int] = None,
    use_time_based_seed_when_unset: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """
    ``DataLoader`` over :class:`SarSim0StreamingDataset` (each sample is one full batch).

    Use ``for batch in loader:`` where ``batch`` is the dict from the dataset (when
    ``num_workers=0``). With ``num_workers>0``, default collate stacks tensors —
    pass ``batch_size=None`` and typically a custom collate if you need the dict
    unchanged per worker batch.

    For a single-process infinite stream (one dict per step), use ``num_workers=0``
    and ``batch_size=None`` (default below).
    """
    ds = SarSim0StreamingDataset(
        context_length,
        prediction_length,
        batch_size,
        output_patch_size=output_patch_size,
        multivariate_prob=multivariate_prob,
        config=config,
        device=device,
        seed=seed,
        reseed_interval=reseed_interval,
        use_time_based_seed_when_unset=use_time_based_seed_when_unset,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )
