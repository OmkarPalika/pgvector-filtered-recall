#!/usr/bin/env python3
"""Rebuild the HNSW index N times and re-measure the same cells on each build.

Every other script in this repo builds the index once. That leaves an open question
about all of its latency numbers: pgvector's parallel build is nondeterministic, so
two builds over byte-identical data produce two different graphs. Any latency
difference attributed to a setting could just be a difference between graphs.

Only the index is rebuilt. The table is untouched, so ground truth is computed once
before the loop and shared by every build — the exact scan does not depend on the
graph, and recomputing it per build would only add noise to the comparison.

Btree indexes on the filter columns are dropped first. With one present the planner
takes btree+sort at 0.1% and the HNSW plan assertion fails, which is what
bench_btree.py exists to show. Rerun bench_btree.py to put them back.
"""
import argparse
import csv
import statistics
import time

from bench import DATA, K, assert_plan, recall, set_guc
from bench_corr import COLUMNS, CONFIGS, LAYOUTS, anchor_sets, arms_for, query_for

BUILDS = 5
NQ = 100
PASSES = 2  # the control: see summarize()
WARMUP = 20  # a fresh build starts with a cold cache; this is not enough to fully
             # warm a 1.3GB index, but it is the same for every build

# The cells worth repeating: the three slowest recipe cells, where a build-to-build
# latency difference would be large enough to mistake for a real effect, plus two
# cells held for their recall rather than their latency.
CELLS = [
    ("bucket",       "near", 1,  "recipe"),
    ("bucket_corr",  "far",  10, "recipe"),
    ("bucket_multi", "far",  10, "recipe"),
    ("bucket_corr",  "near", 1,  "recipe"),
    ("bucket_multi", "near", 1,  "default"),
]


def label(col, arm, sel, cfg):
    return f"{LAYOUTS[col]}/{arm}/{sel / 10.0}%/{cfg}"


def rebuild(cur):
    cur.execute("DROP INDEX IF EXISTS items_embedding_idx")
    cur.execute("SET maintenance_work_mem = '512MB'")
    t = time.perf_counter()
    cur.execute("CREATE INDEX items_embedding_idx ON items USING hnsw "
                "(embedding vector_l2_ops) WITH (m = 16, ef_construction = 64)")
    return time.perf_counter() - t


def spread(vals):
    return round(max(vals) / min(vals), 2) if min(vals) else None


def summarize(rows):
    """Across-build spread, against a within-build control.

    Builds run one after another, so anything that drifts with wall-clock time — host
    load, cache state, thermal throttling — lands on whichever build was running and
    looks exactly like build nondeterminism. Each build is therefore measured twice,
    separated by a full sweep of the other cells, and the two spreads are reported
    side by side:

      within_spread  same graph, different moment    -> everything except the build
      across_spread  different graph, per-build median of the passes

    across >> within is the only evidence that the build itself is responsible. If
    they are comparable, the variance is the host and the build is exonerated.
    """
    by = {}
    for r in rows:
        by.setdefault(r["cell"], {}).setdefault(r["build"], []).append(r)
    out = []
    for cell, builds in by.items():
        per_build = {b: statistics.median(x["p50_ms"] for x in rs)
                     for b, rs in builds.items()}
        p = list(per_build.values())
        rec = [x["recall_mean"] for rs in builds.values() for x in rs]
        within = [spread([x["p50_ms"] for x in rs])
                  for rs in builds.values() if len(rs) > 1]
        out.append({
            "cell": cell,
            "builds": len(builds),
            "passes_per_build": max(len(rs) for rs in builds.values()),
            "p50_min_ms": round(min(p), 2),
            "p50_median_ms": round(statistics.median(p), 2),
            "p50_max_ms": round(max(p), 2),
            "across_spread": spread(p),
            "within_spread_max": max(within) if within else None,
            "recall_min": round(min(rec), 4),
            "recall_max": round(max(rec), 4),
            "recall_spread": round(max(rec) - min(rec), 4),
        })
    return out


def main(dsn, out, builds, nq):
    import h5py
    import numpy as np
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")
        n_rows = cur.execute("SELECT count(*) FROM items").fetchone()[0]
        for col in COLUMNS:
            cur.execute(f"DROP INDEX IF EXISTS items_{col}_idx")
        print(f"{n_rows} rows, btree indexes on filter columns dropped")

        with h5py.File(DATA) as f:
            test = f["test"][:]
            anchors = anchor_sets(f, n_rows)

        # Ground truth once. Identical for every build by construction.
        set_guc(cur, "enable_indexscan", "off")
        set_guc(cur, "enable_bitmapscan", "off")
        set_guc(cur, "enable_seqscan", "on")
        truths, queries = {}, {}
        t = time.perf_counter()
        for col, arm, sel, cfg in CELLS:
            key = (col, arm, sel)
            if key in truths:
                continue
            q_sql = query_for(col)
            qs = arms_for(col, test, anchors, nq)[arm]
            assert_plan(cur, sel, qs[0], "Seq Scan", f"ground truth {key}", q_sql)
            truths[key] = [[r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                           for q in qs]
            queries[key] = qs
        print(f"ground truth for {len(truths)} cells in {time.perf_counter() - t:.1f}s")
        set_guc(cur, "enable_indexscan", "on")
        set_guc(cur, "enable_bitmapscan", "on")
        set_guc(cur, "enable_seqscan", "off")

        rows = []
        for build in range(1, builds + 1):
            secs = rebuild(cur)
            print(f"\nbuild {build}/{builds}: {secs:.1f}s")

            # Passes are the outer loop, so the repeat of a cell is separated from its
            # first measurement by every other cell rather than following it directly.
            for p in range(1, PASSES + 1):
                for col, arm, sel, cfg in CELLS:
                    q_sql = query_for(col)
                    qs, truth = queries[(col, arm, sel)], truths[(col, arm, sel)]
                    for name, val in CONFIGS[cfg].items():
                        set_guc(cur, name, val)
                    lab = label(col, arm, sel, cfg)
                    assert_plan(cur, sel, qs[0], "items_embedding_idx", lab, q_sql)

                    for q in qs[:WARMUP]:
                        cur.execute(q_sql, (sel, q, K)).fetchall()

                    recalls, lats, counts = [], [], []
                    for q, tr in zip(qs, truth):
                        t = time.perf_counter()
                        got = [r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                        lats.append((time.perf_counter() - t) * 1000)
                        recalls.append(recall(got, tr))
                        counts.append(len(got))

                    r = {
                        "build": build,
                        "pass": p,
                        "build_secs": round(secs, 1),
                        "cell": lab,
                        "column": col,
                        "layout": LAYOUTS[col],
                        "arm": arm,
                        "selectivity_pct": sel / 10.0,
                        "config": cfg,
                        "recall_mean": round(float(np.mean(recalls)), 4),
                        "rows_returned_mean": round(float(np.mean(counts)), 2),
                        "p50_ms": round(float(np.percentile(lats, 50)), 2),
                        "p95_ms": round(float(np.percentile(lats, 95)), 2),
                        "n_queries": len(qs),
                    }
                    rows.append(r)
                    print(f"  pass {p} {lab:34s} recall={r['recall_mean']:.3f} "
                          f"rows={r['rows_returned_mean']:5.2f} "
                          f"p50={r['p50_ms']:8.2f}ms p95={r['p95_ms']:9.2f}ms")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    print(f"\nacross {builds} builds x {PASSES} passes "
          f"(across_spread only means something if it exceeds within_spread):")
    print(f"  {'cell':34s} {'p50 min':>9s} {'med':>9s} {'max':>9s} "
          f"{'across':>7s} {'within':>7s} {'recall':>15s}")
    for s in summary:
        print(f"  {s['cell']:34s} {s['p50_min_ms']:9.2f} {s['p50_median_ms']:9.2f} "
              f"{s['p50_max_ms']:9.2f} {s['across_spread']:7.2f} "
              f"{s['within_spread_max']:7.2f} "
              f"{s['recall_min']:7.3f}-{s['recall_max']:.3f} "
              f"(+-{s['recall_spread']:.3f})")

    sum_out = out.replace(".csv", "_summary.csv")
    with open(sum_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {out} ({len(rows)} cells) and {sum_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results_builds.csv")
    p.add_argument("--builds", type=int, default=BUILDS)
    p.add_argument("--nq", type=int, default=NQ)
    a = p.parse_args()
    main(a.dsn, a.out, a.builds, a.nq)
