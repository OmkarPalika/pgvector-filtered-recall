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

    print("ok")


if __name__ == "__main__":
    main()
