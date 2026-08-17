#!/usr/bin/env python3
"""Chart recall + p95 latency vs filter selectivity, one line per iterative_scan mode."""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(src, out, ef):
    series = defaultdict(list)
    with open(src) as f:
        for r in csv.DictReader(f):
            if int(r["ef_search"]) != ef:
                continue
            series[r["iterative_scan"]].append(
                (float(r["selectivity_pct"]), float(r["recall_mean"]), float(r["p95_ms"]))
            )

    if not series:
        raise SystemExit(f"no rows with ef_search={ef} in {src}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for mode, points in series.items():
        points.sort()
        x = [p[0] for p in points]
        ax1.plot(x, [p[1] for p in points], marker="o", label=mode)
        ax2.plot(x, [p[2] for p in points], marker="o", label=mode)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("filter selectivity (% of rows kept)")
        ax.grid(alpha=0.3)
        ax.legend(title="hnsw.iterative_scan")

    ax1.set_ylabel("recall@10")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Recall vs filter selectivity (ef_search={ef})")

    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_title(f"p95 latency vs filter selectivity (ef_search={ef})")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="results.csv")
    p.add_argument("--out", default="recall.png")
    p.add_argument("--ef", type=int, default=40)
    a = p.parse_args()
    main(a.src, a.out, a.ef)
