#!/usr/bin/env python3
"""Compare a uniform filter attribute against a correlated one.

Every other measurement in this repo filters on `bucket`, which is uniform random and
independent of vector position. Real filter attributes are not: one tenant's documents,
one category, one date range all sit together in embedding space. `bucket_corr` models
that by ranking rows on distance to a fixed anchor, so `bucket_corr < N` is a compact
ball holding exactly the same N/1000 of the table.

Where the query sits relative to that ball decides everything, so both cases are run:

  near — query vectors closest to the anchor, i.e. searching inside your own tenant.
         This is the common production shape.
  far  — query vectors furthest from the anchor, i.e. the filtered region is nowhere
         near where the graph walk starts.
"""
import argparse
import csv
import sys
import time

from bench import DATA, K, assert_plan, recall, set_guc, to_literal

COLUMNS = ["bucket", "bucket_corr"]
ARMS = ["near", "far"]
SELECTIVITIES = [10, 1]  # 1% and 0.1%
NQ = 100

CONFIGS = {
    "default": {"hnsw.iterative_scan": "off", "hnsw.ef_search": 40,
                "hnsw.max_scan_tuples": 20_000, "hnsw.scan_mem_multiplier": 1},
    "recipe": {"hnsw.iterative_scan": "relaxed_order", "hnsw.ef_search": 40,
               "hnsw.max_scan_tuples": 100_000, "hnsw.scan_mem_multiplier": 4},
}


def query_for(col):
    return (f"SELECT id FROM items WHERE {col} < %s "
            "ORDER BY embedding <-> %s::vector LIMIT %s")


def main(dsn, out):
    import h5py
    import numpy as np
    import psycopg

    with h5py.File(DATA) as f:
        anchor = f["train"][0]                      # same anchor load.py ranks against
        test = f["test"][:]
    d = ((test - anchor) ** 2).sum(1)
    order = np.argsort(d)
    arms = {"near": [to_literal(test[i]) for i in order[:NQ]],
            "far": [to_literal(test[i]) for i in order[-NQ:]]}

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")
        if not cur.execute("SELECT 1 FROM information_schema.columns WHERE "
                           "table_name='items' AND column_name='bucket_corr'").fetchone():
            sys.exit("items.bucket_corr missing — rerun load.py")

        rows = []
        for col in COLUMNS:
            q_sql = query_for(col)
            for arm in ARMS:
                queries = arms[arm]
                for sel in SELECTIVITIES:
                    pct = sel / 10.0

                    set_guc(cur, "enable_seqscan", "on")
                    set_guc(cur, "enable_indexscan", "off")
                    set_guc(cur, "enable_bitmapscan", "off")
                    assert_plan(cur, sel, queries[0], "Seq Scan", "ground truth", q_sql)
                    truths = [[r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                              for q in queries]
                    set_guc(cur, "enable_indexscan", "on")
                    set_guc(cur, "enable_bitmapscan", "on")
                    set_guc(cur, "enable_seqscan", "off")

                    for cfg_name, cfg in CONFIGS.items():
                        for name, val in cfg.items():
                            set_guc(cur, name, val)
                        assert_plan(cur, sel, queries[0], "items_embedding_idx",
                                    f"{col}/{arm}/{pct}%/{cfg_name}", q_sql)

                        for q in queries[:5]:
                            cur.execute(q_sql, (sel, q, K)).fetchall()

                        recalls, lats, counts = [], [], []
                        for q, truth in zip(queries, truths):
                            t = time.perf_counter()
                            got = [r[0] for r in cur.execute(q_sql, (sel, q, K)).fetchall()]
                            lats.append((time.perf_counter() - t) * 1000)
                            recalls.append(recall(got, truth))
                            counts.append(len(got))

                        r = {
                            "column": col,
                            "layout": "uniform" if col == "bucket" else "correlated",
                            "arm": arm,
                            "selectivity_pct": pct,
                            "config": cfg_name,
                            "recall_mean": round(float(np.mean(recalls)), 4),
                            "rows_returned_mean": round(float(np.mean(counts)), 2),
                            "p50_ms": round(float(np.percentile(lats, 50)), 2),
                            "p95_ms": round(float(np.percentile(lats, 95)), 2),
                            "n_queries": len(queries),
                        }
                        rows.append(r)
                        print(f"  {r['layout']:10s} {arm:4s} {pct:5.1f}% {cfg_name:8s} "
                              f"recall={r['recall_mean']:.3f} "
                              f"rows={r['rows_returned_mean']:5.2f} "
                              f"p50={r['p50_ms']:7.2f}ms p95={r['p95_ms']:8.2f}ms")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results_corr.csv")
    a = p.parse_args()
    main(a.dsn, a.out)
