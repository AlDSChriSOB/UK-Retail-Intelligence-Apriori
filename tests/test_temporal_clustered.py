import numpy as np
from src.temporal_clustered import paired_cluster_interval

def test_cluster_bootstrap_identical_pairs():
    diffs = np.zeros(6)
    ci = paired_cluster_interval(diffs, ["a","a","b","c","c","c"], 100, 3)
    assert ci == [0.0, 0.0]

def test_cluster_bootstrap_constant_positive():
    diffs = np.ones(4)
    ci = paired_cluster_interval(diffs, ["a","a","b","c"], 100, 3)
    assert ci == [1.0, 1.0]

def test_cluster_bootstrap_deterministic():
    diffs = np.array([1.,0.,-1.,1.])
    clusters = ["a","a","b","c"]
    assert paired_cluster_interval(diffs, clusters, 100, 42) == paired_cluster_interval(diffs, clusters, 100, 42)
