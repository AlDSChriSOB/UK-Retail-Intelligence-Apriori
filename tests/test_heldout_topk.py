import numpy as np
import pytest
from src.heldout_topk import recommend

def test_rule_rank_and_exclusion():
    popularity = np.array([10, 8, 6, 4])
    rules = {0: [(1, .8), (2, .4)]}
    assert recommend({0}, popularity, rules, 3, "rules") == [1, 2, 3]

def test_popularity_excludes_observed():
    assert recommend({0}, np.array([10, 8, 6]), {}, 2, "popularity") == [1, 2]

def test_random_seed_reproducible():
    p = np.array([2, 3, 4, 5])
    a = recommend({0}, p, {}, 3, "random", np.random.default_rng(7))
    b = recommend({0}, p, {}, 3, "random", np.random.default_rng(7))
    assert a == b and 0 not in a

def test_no_train_unseen_fallback():
    assert recommend({0}, np.array([4, 3, 0]), {}, 10, "popularity") == [1]
