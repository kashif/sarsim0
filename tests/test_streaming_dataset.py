"""Tests for vectorized streaming dataset (Chronos-style batches)."""

from typing import Any

import torch
from sarsim0 import SarSimConfig
from sarsim0.streaming_dataset import (
    SarSim0StreamingDataset,
    build_mixed_covariate_batch,
    build_vectorized_mixed_batch,
    create_streaming_dataloader,
)


def _tiny_config(**kwargs: Any) -> SarSimConfig:
    config = SarSimConfig(
        series_length=128,
        context_window=32,
        prediction_window=16,
        burn_in=40,
        multivariate_prob=0.4,
        n_variates_range=(2, 3),
        sarima2_prob=0.0,
        with_covariates_prob=0.0,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


class TestBuildVectorizedMixedBatch:
    def test_keys_shapes_and_num_patches(self, generator):
        cfg = _tiny_config()
        patch = 8
        b = build_vectorized_mixed_batch(
            12,
            cfg.context_window,
            cfg.prediction_window,
            cfg,
            generator,
            torch.device("cpu"),
            output_patch_size=patch,
        )
        assert set(b) == {
            "context",
            "future_target",
            "future_covariates",
            "group_ids",
            "num_output_patches",
        }
        assert b["num_output_patches"] == (cfg.prediction_window + patch - 1) // patch
        assert b["context"].shape[1] == cfg.context_window
        assert b["future_target"].shape[1] == cfg.prediction_window
        assert b["future_target"].shape == b["future_covariates"].shape
        assert torch.isnan(b["future_covariates"]).all()
        assert b["group_ids"].dtype == torch.long
        assert torch.isfinite(b["context"]).all()
        assert torch.isfinite(b["future_target"]).all()

    def test_univariate_only_row_count_matches_batch_size(self, generator):
        cfg = _tiny_config(multivariate_prob=0.0)
        b = build_vectorized_mixed_batch(
            7,
            cfg.context_window,
            cfg.prediction_window,
            cfg,
            generator,
            torch.device("cpu"),
        )
        assert b["context"].shape[0] == 7
        assert b["future_target"].shape[0] == 7
        torch.testing.assert_close(b["group_ids"], torch.arange(7))

    def test_multivariate_only_groups_share_group_id(self, generator):
        cfg = _tiny_config(multivariate_prob=1.0, n_variates_range=(3, 3))
        b = build_vectorized_mixed_batch(
            4,
            cfg.context_window,
            cfg.prediction_window,
            cfg,
            generator,
            torch.device("cpu"),
        )
        assert b["context"].shape[0] == 4 * 3
        g = b["group_ids"]
        for gid in range(4):
            mask = g == gid
            assert mask.sum().item() == 3


class TestSarSim0StreamingDataset:
    def test_iter_yields_consistent_dict(self):
        ds = SarSim0StreamingDataset(
            context_length=32,
            prediction_length=8,
            batch_size=5,
            seed=123,
            multivariate_prob=0.0,
            config=_tiny_config(multivariate_prob=0.0),
        )
        it = iter(ds)
        b = next(it)
        assert b["context"].shape[0] == 5
        assert 8 <= b["context"].shape[1] <= 32
        assert b["future_target"].shape == (5, 8)

    def test_create_streaming_dataloader(self):
        dl = create_streaming_dataloader(
            24,
            8,
            batch_size=4,
            seed=0,
            num_workers=0,
            multivariate_prob=0.0,
        )
        batch = next(iter(dl))
        assert batch["context"].shape[0] == 4

    def test_covariate_streaming_batches_match_chronos2_contract(self):
        cfg = _tiny_config(
            multivariate_prob=1.0,
            with_covariates_prob=1.0,
            n_variates_range=(2, 2),
            n_past_covariates_range=(1, 1),
            n_future_covariates_range=(1, 1),
        )
        ds = SarSim0StreamingDataset(
            context_length=32,
            prediction_length=8,
            batch_size=3,
            seed=123,
            config=cfg,
        )

        batch = next(iter(ds))
        assert set(batch) == {
            "context",
            "future_target",
            "future_covariates",
            "group_ids",
            "num_output_patches",
        }
        assert 8 <= batch["context"].shape[1] <= 32
        assert batch["future_target"].shape[1] == 8
        assert batch["future_covariates"].shape == batch["future_target"].shape
        assert batch["group_ids"].dtype == torch.long

        # 2 targets + 1 past-only covariate + 1 known-future covariate per group
        for gid in range(3):
            mask = batch["group_ids"] == gid
            assert mask.sum().item() == 4
            group_future_target = batch["future_target"][mask]
            group_future_covs = batch["future_covariates"][mask]

            assert torch.isfinite(group_future_target[:2]).all()
            assert torch.isnan(group_future_target[2:]).all()
            assert torch.isnan(group_future_covs[:3]).all()
            assert torch.isfinite(group_future_covs[3]).all()


class TestCovariateBatchBuilder:
    def test_build_mixed_covariate_batch_multivariate(self, generator):
        cfg = _tiny_config(
            multivariate_prob=1.0,
            with_covariates_prob=1.0,
            n_variates_range=(3, 3),
            n_past_covariates_range=(1, 1),
            n_future_covariates_range=(2, 2),
        )
        batch = build_mixed_covariate_batch(
            2,
            cfg.context_window,
            cfg.prediction_window,
            cfg,
            generator,
            output_patch_size=4,
        )

        assert batch["num_output_patches"] == 4
        assert batch["context"].shape[0] == 2 * (3 + 1 + 2)
        assert batch["future_target"].shape == batch["future_covariates"].shape
        for gid in range(2):
            mask = batch["group_ids"] == gid
            group_future_target = batch["future_target"][mask]
            group_future_covs = batch["future_covariates"][mask]
            assert torch.isfinite(group_future_target[:3]).all()
            assert torch.isnan(group_future_target[3:]).all()
            assert torch.isnan(group_future_covs[:4]).all()
            assert torch.isfinite(group_future_covs[4:]).all()
