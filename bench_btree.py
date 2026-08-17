#!/usr/bin/env python3
"""Does a btree index on the filter column beat tuning HNSW?

The zero-row failure is not really about HNSW. With no index on the filter column the
planner's only alternative to the vector index is a 1M-row seq scan, which it rightly
refuses. Given a btree it can read the filtered rows directly and sort them exactly,
which cannot miss anything — there is no recall question left. That is also why the
100k run never showed the failure: a seq scan was cheap enough there that the planner
escaped on its own.

An existence probe (`SELECT 1 FROM items WHERE filter LIMIT 1`) does not work as a
predictor, which is worth recording because it is the obvious first idea. In the failing
case the predicate matches thousands of rows — they are simply nowhere near the query —
so the probe returns a row immediately and predicts nothing.

Two results here are independent of machine load, which matters because this was written
while the host was saturated:
  * which plan the planner chooses, decided by the cost model rather than the clock
  * whether the query still returns zero rows
Latency is recorded but should be re-read on an idle host before it is quoted.
"""
import argparse
import csv
import time

from bench import DATA, K, recall, set_guc, to_literal
from bench_corr import COLUMNS, LAYOUTS, anchor_sets, arms_for, query_for

ARMS = ["near", "far"]
SELECTIVITIES = [1, 10, 100]  # 0.1%, 1%, 10%
NQ = 50                       # ground truth dominates the runtime here

RECIPE = {"hnsw.iterative_scan": "relaxed_order", "hnsw.ef_search": 40,
          "hnsw.max_scan_tuples": 100_000, "hnsw.scan_mem_multiplier": 4}


def plan_label(plan):
    if "items_embedding_idx" in plan:
        return "hnsw"
    if "Bitmap" in plan:
        return "bitmap+sort"
    if "Index Scan" in plan or "Index Only Scan" in plan:
        return "btree+sort"
    return "seqscan+sort"


def main(dsn, out):
    import h5py
    import numpy as np
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")
        n_rows = cur.execute("SELECT count(*) FROM items").fetchone()[0]

        for col in COLUMNS:
            t = time.perf_counter()
            cur.execute(f"CREATE INDEX IF NOT EXISTS items_{col}_idx ON items ({col})")
            print(f"btree on {col}: {time.perf_counter() - t:.1f}s")
        cur.execute("ANALYZE items")

        with h5py.File(DATA) as f:
            test = f["test"][:]
            anchors = anchor_sets(f, n_rows)

        rows = []
        for col in COLUMNS:
            q_sql = query_for(col)
            arms = arms_for(col, test, anchors, NQ)
            for arm in ARMS:
                queries = arms[arm]
                for sel in SELECTIVITIES:
                    pct = sel / 10.0

                    set_guc(cur, "enable_seqscan", "on")
                    set_guc(cur, "enable_indexscan", "off")
                    set_guc(cur, "enable_bitmapscan", "off")
                    truths = [[r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                              for q in queries]
                    set_guc(cur, "enable_indexscan", "on")
                    set_guc(cur, "enable_bitmapscan", "on")

                    for cfg_name in ("planner_free", "hnsw_recipe"):
                        for name, val in RECIPE.items():
                            set_guc(cur, name, val)
                        # planner_free: btree is available and nothing is forced, so the
                        # cost model decides. hnsw_recipe: the vector index is forced,
                        # which is what every earlier measurement in this repo did.
                        set_guc(cur, "enable_seqscan",
                                "on" if cfg_name == "planner_free" else "off")
                        if cfg_name == "hnsw_recipe":
                            set_guc(cur, "enable_bitmapscan", "off")
                            set_guc(cur, "enable_indexscan", "on")

                        plan = "\n".join(r[0] for r in cur.execute(
                            "EXPLAIN " + q_sql, (sel, queries[0], K)).fetchall())
                        chosen = plan_label(plan)

                        for q in queries[:5]:
                            cur.execute(q_sql, (sel, q, K)).fetchall()

                        recalls, lats, counts = [], [], []
                        for q, truth in zip(queries, truths):
                            t = time.perf_counter()
                            got = [r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                            lats.append((time.perf_counter() - t) * 1000)
                            recalls.append(recall(got, truth))
                            counts.append(len(got))

                        set_guc(cur, "enable_bitmapscan", "on")

                        r = {
                            "layout": LAYOUTS[col],
                            "arm": arm,
                            "selectivity_pct": pct,
                            "config": cfg_name,
                            "plan": chosen,
                            "recall_mean": round(float(np.mean(recalls)), 4),
                            "rows_returned_mean": round(float(np.mean(counts)), 2),
                            "p50_ms": round(float(np.percentile(lats, 50)), 2),
                            "n_queries": len(queries),
                        }
                        rows.append(r)
                        print(f"  {r['layout']:12s} {arm:4s} {pct:5.1f}% "
                              f"{cfg_name:12s} {chosen:12s} "
                              f"recall={r['recall_mean']:.3f} "
                              f"rows={r['rows_returned_mean']:5.2f} "
                              f"p50={r['p50_ms']:8.2f}ms")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results_btree.csv")
    a = p.parse_args()
    main(a.dsn, a.out)
