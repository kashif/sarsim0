"""Tests for SarSim0 pipeline."""

import torch
from sarsim0.pipeline import (
    SarSim0Dataset,
    SarSim0Generator,
    create_dataloader,
    generate_sarsim0_batch,
)


class TestGenerateSarsim0Batch:
    def test_output_shape_and_finite(self, config, generator):
        batch_size, length = 16, 1000
        y = generate_sarsim0_batch(batch_size, length, config, generator)
        assert y.shape == (batch_size, length - config.burn_in)
        assert y.dtype == torch.float32
        assert torch.isfinite(y).all()

    def test_sarima2_only_path_runs(self, config, generator):
        config.sarima2_prob = 1.0
        y = generate_sarsim0_batch(6, 400, config, generator)
        assert y.shape[0] == 6
        assert torch.isfinite(y).all()


class TestSarSim0Dataset:
    def test_finite_batches_and_shapes(self, small_config):
        dataset = SarSim0Dataset(
            batch_size=16,
            config=small_config,
            seed=42,
            num_batches=3,
        )
        batches = list(dataset)
        assert len(batches) == 3
        for context, target in batches:
            assert context.shape == (16, small_config.context_window)
            assert target.shape == (16, small_config.prediction_window)
            assert torch.isfinite(context).all() and torch.isfinite(target).all()

    def test_infinite_iteration_breaks_cleanly(self, small_config):
        dataset = SarSim0Dataset(
            batch_size=4,
            config=small_config,
            seed=42,
            num_batches=None,
        )
        count = 0
        for _ in dataset:
            count += 1
            if count == 5:
                break
        assert count == 5


class TestSarSim0Generator:
    def test_generate_batch_shapes_and_variation(self, small_config):
        gen = SarSim0Generator(config=small_config, seed=42)
        context, target = gen.generate_batch(batch_size=16)
        assert context.shape == (16, small_config.context_window)
        assert target.shape == (16, small_config.prediction_window)
        assert torch.isfinite(context).all() and torch.isfinite(target).all()
        ctx2, _ = gen.generate_batch(batch_size=16)
        assert not torch.allclose(context, ctx2)

    def test_generate_series_shapes(self, small_config):
        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_series(batch_size=8, length=500)
        assert y.shape == (8, 500)
        assert torch.isfinite(y).all()

    def test_default_config_batch(self):
        gen = SarSim0Generator(seed=42)
        context, target = gen.generate_batch(batch_size=4)
        assert context.shape == (4, 4096)
        assert target.shape == (4, 512)
        assert torch.isfinite(context).all() and torch.isfinite(target).all()


class TestCreateDataloader:
    def test_iteration_shapes_and_finite(self, small_config):
        dl = create_dataloader(
            batch_size=16,
            config=small_config,
            num_workers=0,
            seed=42,
            num_batches_per_epoch=3,
        )
        batches = list(dl)
        assert len(batches) == 3
        for context, target in batches:
            assert context.shape == (16, small_config.context_window)
            assert target.shape == (16, small_config.prediction_window)
            assert torch.isfinite(context).all() and torch.isfinite(target).all()


class TestIntegration:
    def test_large_batch(self, small_config):
        gen = SarSim0Generator(config=small_config, seed=42)
        context, target = gen.generate_batch(batch_size=256)
        assert context.shape == (256, small_config.context_window)
        assert torch.isfinite(context).all()
