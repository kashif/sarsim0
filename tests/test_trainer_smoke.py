"""Smoke tests for training on streaming Chronos-2-style batches."""

from __future__ import annotations

import torch
from sarsim0 import SarSimConfig
from sarsim0.streaming_dataset import SarSim0StreamingDataset
from torch.utils.data import DataLoader


class DummyChronos2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 1)

    def forward(self, context, future_target, future_covariates, **kwargs):
        context_mask = torch.isfinite(context)
        context_mean = torch.nan_to_num(context, nan=0.0).sum(dim=-1) / context_mask.sum(
            dim=-1
        ).clamp_min(1)

        cov_mask = torch.isfinite(future_covariates)
        cov_mean = torch.nan_to_num(future_covariates, nan=0.0).sum(dim=-1) / cov_mask.sum(
            dim=-1
        ).clamp_min(1)

        features = torch.stack([context_mean, cov_mean], dim=-1)
        pred = self.head(features).expand(-1, future_target.shape[-1])

        target_mask = torch.isfinite(future_target)
        target = torch.nan_to_num(future_target, nan=0.0)
        return ((pred - target) ** 2)[target_mask].mean()


def _make_dataset() -> SarSim0StreamingDataset:
    config = SarSimConfig(
        series_length=96,
        burn_in=32,
        context_window=24,
        prediction_window=8,
        sarima2_prob=0.0,
        multivariate_prob=1.0,
        with_covariates_prob=1.0,
        n_variates_range=(2, 2),
        n_past_covariates_range=(1, 1),
        n_future_covariates_range=(1, 1),
    )
    return SarSim0StreamingDataset(
        context_length=24,
        prediction_length=8,
        batch_size=2,
        config=config,
        seed=123,
    )


def test_streaming_batches_support_few_training_steps():
    dataset = _make_dataset()
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    model = DummyChronos2Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for step, batch in zip(range(3), loader):
        optimizer.zero_grad()
        loss = model(
            context=batch["context"],
            future_target=batch["future_target"],
            future_covariates=batch["future_covariates"],
            group_ids=batch["group_ids"],
            num_output_patches=batch["num_output_patches"],
        )
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))
