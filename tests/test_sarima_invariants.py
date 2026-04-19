"""Regression tests for paper-style SARIMA sampling invariants."""

from unittest import mock

import numpy as np
import sarsim0.sarima as sarima_mod
import torch
from sarsim0 import SarSimConfig
from sarsim0.sarima import generate_sarima_batch
from sarsim0.sarima2 import generate_sarima2_paired_batch


class TestStabilityMixture:
    """Section 4.1: each draw must zero either non-seasonal or seasonal AR coefficients."""

    def test_each_vectorized_subbatch_has_ar_or_seasonal_ar_zero_order(self, config):
        recorded = []

        def side_effect(batch_size, length, ar, ma, sar, sma, s, d, D, rng):
            recorded.append((len(ar), len(sar)))
            return np.zeros((batch_size, length), dtype=np.float64)

        with mock.patch.object(sarima_mod, "generate_sarima_vectorized", side_effect=side_effect):
            g = torch.Generator().manual_seed(101)
            generate_sarima_batch(80, 300, config, g)

        assert recorded, "expected at least one vectorized sub-batch"
        for nar, nsar in recorded:
            assert nar == 0 or nsar == 0, (
                "expected p=0 or P=0 after mixture; "
                f"got non-seasonal AR coeff count {nar}, seasonal AR coeff count {nsar}"
            )


class TestSarima2PairedSeasonalities:
    """SARIMA-2 uses one (s_base, s_env) pair from config per batch call."""

    def test_paired_batch_passes_fixed_base_and_envelope_periods(self):
        pair = (12, 7)
        cfg = SarSimConfig(
            seasonality_pairs=[pair],
            series_length=200,
            burn_in=20,
            sarima2_prob=0.0,
        )
        seen_base = []
        seen_env = []

        def fake_gen(bs, length, c, gen, device, fixed_s=None):
            seen_base.append(fixed_s)
            return torch.zeros(bs, length, dtype=torch.float32, device=device), None

        def fake_apply(y, c, gen, device, envelope_s=None):
            seen_env.append(envelope_s)
            return y, torch.zeros(y.shape[0], dtype=torch.long, device=device)

        with (
            mock.patch("sarsim0.sarima2.generate_sarima_batch", side_effect=fake_gen),
            mock.patch("sarsim0.sarima2.apply_sarima2_vectorized", side_effect=fake_apply),
        ):
            generate_sarima2_paired_batch(
                4, 100, cfg, torch.Generator().manual_seed(0), torch.device("cpu")
            )

        assert seen_base == [pair[0]]
        assert seen_env == [pair[1]]
