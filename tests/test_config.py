"""Tests for configuration defaults."""

from sarsim0 import SarSimConfig


def test_default_seasonality_pairs_match_paper_table_9():
    config = SarSimConfig()
    assert config.seasonality_pairs == [
        (24, 7),
        (7, 52),
        (0, 7),
        (0, 4),
        (0, 24),
        (0, 52),
    ]
