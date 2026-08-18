#!/usr/bin/env python3
"""Measure pgvector recall + latency vs filter selectivity, ef_search, iterative_scan.

Ground truth is an exact scan in the same database (index scans disabled), so there
is no second pipeline that can disagree with the one under test.

Two things are verified at runtime rather than assumed, because both fail silently:
  * every GUC is read back with SHOW after being set (see set_guc)
  * every query's plan is checked to be the scan type this measurement requires
"""
import argparse
import csv
import os
import sys
import time

# h5py / numpy / psycopg are imported inside main() so test_bench.py can check the
# metric with no third-party packages installed and nothing downloaded.

DATA = os.path.join("data", "sift-128-euclidean.hdf5")

SELECTIVITIES = [1000, 500, 100, 10, 1]  # WHERE bucket < N  ->  N/1000
MODES = ["off", "relaxed_order", "strict_order"]
EFS = [40, 100, 400]
K = 10
NQ = 200

QUERY = "SELECT id FROM items WHERE bucket < %s ORDER BY embedding <-> %s::vector LIMIT %s"


def recall(got, truth):
    """Fraction of the true top-k that the index actually returned.

    Denominator is len(truth), not K: when the filtered subset holds fewer than K
    rows, an exact scan returns fewer than K and the index cannot beat that.
    """
    if not truth:
        return 1.0
    return len(set(got) & set(truth)) / len(truth)


def to_literal(vec):
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def set_guc(cur, name, value):
    """Set a GUC and prove it took effect.

    Postgres accepts `SET` on any dotted name it does not recognise, storing a
    placeholder and raising nothing. Before pgvector's library is loaded into the
    session, every hnsw.* setting lands in that hole — the benchmark then sweeps a
    grid of settings that are all silently ignored and reports one configuration
    N times. Reading the value back is the only way to catch it.
    """
    cur.execute("SELECT set_config(%s, %s, false)", (name, str(value)))
    got = cur.execute(f"SHOW {name}").fetchone()[0]
    if str(got) != str(value):
        sys.exit(f"FATAL: {name}={value!r} did not apply — server reports {got!r}")


def warm_cache(cur, query, sel, queries, k=K):
    """Run the whole query set once and throw it away.

    Every sweep here measures several configurations back to back over the same cell.
    A truncated warmup leaves the first configuration in that loop absorbing the cold
    cache, so it reads slower for that reason alone and the ordering shows up in the
    results as if it were an effect of the setting — 143.69ms against 26.90ms between
    two bench_btree.py configurations that chose the identical plan and scored the
    identical recall. Warming with exactly the queries about to be timed is what makes
    the order stop mattering; a merely larger fixed warmup shrinks the bias without
    removing it.
    """
    for q in queries:
        cur.execute(query, (sel, q, k)).fetchall()


def plan_text(cur, sel, qv, query=QUERY):
    return "\n".join(r[0] for r in cur.execute(
        "EXPLAIN " + query, (sel, qv, K)).fetchall())


def assert_plan(cur, sel, qv, must_contain, label, query=QUERY):
    """Fail loudly if the planner is not running the scan this measurement assumes."""
    plan = plan_text(cur, sel, qv, query)
    if must_contain not in plan:
        sys.exit(f"FATAL: {label} expected {must_contain!r} in plan, got:\n{plan}")


def main(dsn, out):
    import h5py
    import numpy as np
    import psycopg

    with h5py.File(DATA) as f:
        queries = [to_literal(v) for v in f["test"][:NQ]]

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()

        # Registers the hnsw.* GUCs. Without this they are unrecognised, and every
        # set_config below would be accepted and discarded.
        cur.execute("LOAD 'vector'")

        row = cur.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if not row:
            sys.exit("pgvector extension not installed — run load.py first")
        if tuple(int(x) for x in row[0].split(".")[:2]) < (0, 8):
            sys.exit(f"pgvector {row[0]} < 0.8.0: hnsw.iterative_scan unavailable")
        print(f"pgvector {row[0]}")
        print(f"defaults: ef_search={cur.execute('SHOW hnsw.ef_search').fetchone()[0]} "
              f"iterative_scan={cur.execute('SHOW hnsw.iterative_scan').fetchone()[0]} "
              f"max_scan_tuples={cur.execute('SHOW hnsw.max_scan_tuples').fetchone()[0]}")

        rows = []
        for sel in SELECTIVITIES:
            pct = sel / 10.0

            set_guc(cur, "enable_seqscan", "on")
            set_guc(cur, "enable_indexscan", "off")
            set_guc(cur, "enable_bitmapscan", "off")
            assert_plan(cur, sel, queries[0], "Seq Scan", "ground truth")
            t = time.perf_counter()
            truths = [[r[0] for r in cur.execute(QUERY, (sel, q, K)).fetchall()]
                      for q in queries]
            gt_secs = time.perf_counter() - t
            set_guc(cur, "enable_indexscan", "on")
            set_guc(cur, "enable_bitmapscan", "on")

            # What the planner picks when left alone. Below roughly 1% selectivity it
            # abandons HNSW for a seq scan + sort, which is exact — so recall looks
            # perfect while the vector index is not being used at all.
            natural = ("hnsw" if "items_embedding_idx" in plan_text(cur, sel, queries[0])
                       else "seqscan")

            # Force the index so the sweep measures HNSW rather than the planner.
            set_guc(cur, "enable_seqscan", "off")

            print(f"\nselectivity {pct}% — ground truth in {gt_secs:.1f}s "
                  f"(mean {np.mean([len(x) for x in truths]):.1f} rows), "
                  f"planner would choose: {natural}")

            for mode in MODES:
                for ef in EFS:
                    set_guc(cur, "hnsw.ef_search", ef)
                    set_guc(cur, "hnsw.iterative_scan", mode)
                    assert_plan(cur, sel, queries[0],
                                "items_embedding_idx", f"{mode}/ef={ef}")

                    warm_cache(cur, QUERY, sel, queries)

                    recalls, lats, counts = [], [], []
                    for q, truth in zip(queries, truths):
                        t = time.perf_counter()
                        got = [r[0] for r in cur.execute(QUERY, (sel, q, K)).fetchall()]
                        lats.append((time.perf_counter() - t) * 1000)
                        recalls.append(recall(got, truth))
                        counts.append(len(got))

                    r = {
                        "selectivity_pct": pct,
                        "bucket_lt": sel,
                        "iterative_scan": mode,
                        "ef_search": ef,
                        "planner_default": natural,
                        "recall_mean": round(float(np.mean(recalls)), 4),
                        "rows_returned_mean": round(float(np.mean(counts)), 2),
                        "p50_ms": round(float(np.percentile(lats, 50)), 2),
                        "p95_ms": round(float(np.percentile(lats, 95)), 2),
                        "truth_rows_mean": round(float(np.mean([len(x) for x in truths])), 1),
                        "n_queries": len(queries),
                    }
                    rows.append(r)
                    print(f"  {mode:14s} ef={ef:<4d} recall={r['recall_mean']:.3f} "
                          f"rows={r['rows_returned_mean']:5.2f} "
                          f"p50={r['p50_ms']:6.2f}ms p95={r['p95_ms']:7.2f}ms")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results.csv")
    a = p.parse_args()
    main(a.dsn, a.out)
