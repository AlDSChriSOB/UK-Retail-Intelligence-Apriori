"""Tests for paired invoice bootstrap."""
import pytest
from src.paired_bootstrap import paired_bootstrap


def test_identical_methods_have_zero_difference():
    result = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], replicates=200, seed=1)
    assert result["difference"] == 0
    assert result["ci95_percentile"] == [0, 0]
    assert result["a_only"] == result["b_only"] == 0


def test_perfect_improvement():
    result = paired_bootstrap([1] * 20, [0] * 20, replicates=200, seed=1)
    assert result["difference"] == 1
    assert result["ci95_percentile"] == [1, 1]
    assert result["a_only"] == 20


def test_reproducible_bootstrap():
    a = [1, 0, 1, 0, 1, 1, 0, 0]
    b = [0, 0, 1, 1, 0, 1, 0, 1]
    assert paired_bootstrap(a, b, 200, 5) == paired_bootstrap(a, b, 200, 5)


def test_invalid_outcomes_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [0, 1], replicates=200)
