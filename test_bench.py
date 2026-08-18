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
    check_multicluster_buckets()
    check_build_summary()
    print("ok")


def check_build_summary():
    """The whole point of the repeated-build run is the spread, so a wrong reduction
    here would fabricate the conclusion it is meant to test."""
    from bench_builds import summarize

    def row(cell, build, p, ms, rec=1.0):
        return {"cell": cell, "build": build, "pass": p,
                "p50_ms": ms, "recall_mean": rec}

    # Cell "a": three builds, each measured twice. Per-build medians are 110, 220, 150,
    # so across_spread is 2.0 while no single build disagreed with itself by more than
    # 1.1x. That gap is the finding the run is built to produce.
    rows = [row("a", 1, 1, 100.0, 0.90), row("a", 1, 2, 120.0, 0.90),
            row("a", 2, 1, 210.0, 0.91), row("a", 2, 2, 230.0, 0.91),
            row("a", 3, 1, 150.0, 0.90), row("a", 3, 2, 150.0, 0.90),
            row("b", 1, 1, 10.0), row("b", 1, 2, 20.0)]
    a, b = summarize(rows)

    assert a["builds"] == 3 and a["passes_per_build"] == 2
    assert (a["p50_min_ms"], a["p50_median_ms"], a["p50_max_ms"]) == (110.0, 150.0, 220.0)
    assert a["across_spread"] == 2.0, "must compare per-build medians, not raw passes"
    assert a["within_spread_max"] == 1.2
    assert a["recall_spread"] == 0.01

    # Cell "b": one build, both passes. Nothing to say about builds — across_spread has
    # to be neutral while within_spread still reports the 2x the host produced, or the
    # summary would credit the build for host noise.
    assert b["builds"] == 1 and b["across_spread"] == 1.0
    assert b["within_spread_max"] == 2.0


def check_multicluster_buckets():
    """Selectivity must stay N/1000 while the region becomes several balls. If the
    per-cluster ranking is wrong the buckets skew and every cell measures a different
    selectivity than it reports."""
    try:
        import numpy as np
    except ImportError:
        return
    from load import BUCKETS, anchor_indices, multicluster_buckets

    rng = np.random.default_rng(1)
    # Four tight, well-separated blobs.
    train = np.vstack([rng.normal(c * 100, 1.0, (2000, 8)) for c in range(4)]
                      ).astype(np.float32)
    multi = multicluster_buckets(train, n_clusters=4)

    counts = np.bincount(multi, minlength=BUCKETS)
    assert counts.min() == counts.max() == len(train) // BUCKETS, f"skewed: {counts}"

    # The selected region must draw from every cluster, not concentrate in one.
    # That is the whole difference from the single-ball attribute.
    blob = np.repeat(np.arange(4), 2000)
    selected = blob[multi < 100]                       # innermost 10%
    assert len(np.unique(selected)) == 4, "region should span all clusters"
    assert np.bincount(selected).min() == 200, "each cluster contributes its own share"

    assert len(anchor_indices(1000, 4)) == 4
    assert len(set(anchor_indices(1000, 4))) == 4, "anchors must be distinct"


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
