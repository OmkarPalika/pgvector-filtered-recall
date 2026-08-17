#!/usr/bin/env python3
"""Self-check for the recall metric. Run: python test_bench.py"""
from bench import recall, to_literal


def main():
    assert recall([1, 2, 3], [1, 2, 3]) == 1.0
    assert recall([9, 8, 7], [1, 2, 3]) == 0.0
    assert recall([1, 2, 9], [1, 2, 3]) == 2 / 3

    # Filtered subset smaller than K: an exact scan returns 2 rows, so returning
    # those 2 is perfect recall. Dividing by K instead would report 0.2 and make
    # every high-selectivity cell look broken.
    assert recall([1, 2], [1, 2]) == 1.0

    # Empty subset: nothing to find, vacuously perfect. Must not divide by zero.
    assert recall([], []) == 1.0

    # Order and duplicates must not affect a set-overlap metric.
    assert recall([3, 1, 2], [1, 2, 3]) == 1.0
    assert recall([1, 1, 1], [1, 2, 3]) == 1 / 3

    assert to_literal([1.0, 2.5]) == "[1,2.5]"

    check_correlated_buckets()
    print("ok")


def check_correlated_buckets():
    """The rank assignment decides selectivity, so a slip here silently changes what
    every correlated cell measures. Skipped rather than failed without numpy, which
    keeps the recall checks above runnable with no third-party packages."""
    try:
        import numpy as np
    except ImportError:
        return
    from load import BUCKETS, correlated_buckets

    # 1-D points at 0..999; the anchor is train[0] == 0, so rank order is just value
    # order and bucket b must contain exactly the points with that rank decile.
    train = np.arange(1000, dtype=np.float32).reshape(-1, 1)
    corr = correlated_buckets(train)
    assert list(corr) == list(range(BUCKETS)), "ranks must map 1:1 onto buckets here"

    # Equal-sized buckets are what make `bucket_corr < N` exactly N/1000 selectivity.
    train = np.random.default_rng(0).random((5000, 4), dtype=np.float32)
    corr = correlated_buckets(train)
    counts = np.bincount(corr, minlength=BUCKETS)
    assert counts.min() == counts.max() == 5000 // BUCKETS, f"uneven buckets: {counts}"

    # Nearest to the anchor must land in bucket 0, farthest in the last bucket.
    d = ((train - train[0]) ** 2).sum(1)
    assert corr[np.argsort(d)[0]] == 0
    assert corr[np.argsort(d)[-1]] == BUCKETS - 1


if __name__ == "__main__":
    main()
