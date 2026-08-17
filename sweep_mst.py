#!/usr/bin/env python3
"""Sweep hnsw.max_scan_tuples and hnsw.scan_mem_multiplier at low selectivity.

At 1M rows, iterative_scan tops out around 0.85 recall at 0.1% selectivity and
does not respond to ef_search. Two documented knobs are left. This asks whether
either of them moves the ceiling, and what it costs.

Reuses bench.py's helpers so the ground truth, the GUC read-back and the plan
assertion are identical to the main grid.
"""
import argparse
import csv
import sys
import time

from bench import DATA, K, QUERY, assert_plan, recall, set_guc, to_literal

SELECTIVITIES = [10, 1]  # 1% and 0.1% — where iterative_scan stops being enough
MSTS = [20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
SMMS = [1, 2]
EFS = [40]  # flat while a cap binds; becomes worth sweeping once both are lifted
MODE = "relaxed_order"
NQ = 100
WARMUP = 5


def main(dsn, out, sels, msts, smms, efs):
    import h5py
    import numpy as np
    import psycopg

    with h5py.File(DATA) as f:
        queries = [to_literal(v) for v in f["test"][:NQ]]

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")

        n = cur.execute("SELECT count(*) FROM items").fetchone()[0]
        print(f"{n} rows")

        rows = []
        for sel in sels:
            pct = sel / 10.0

            set_guc(cur, "enable_seqscan", "on")
            set_guc(cur, "enable_indexscan", "off")
            set_guc(cur, "enable_bitmapscan", "off")
            assert_plan(cur, sel, queries[0], "Seq Scan", "ground truth")
            truths = [[r[0] for r in cur.execute(QUERY, (sel, q, K)).fetchall()]
                      for q in queries]
            set_guc(cur, "enable_indexscan", "on")
            set_guc(cur, "enable_bitmapscan", "on")
            set_guc(cur, "enable_seqscan", "off")

            print(f"\nselectivity {pct}%  (subset ~{int(n * sel / 1000)} rows)")
            set_guc(cur, "hnsw.iterative_scan", MODE)

            for smm, mst, ef in ((s, m, e) for s in smms for m in msts for e in efs):
                    set_guc(cur, "hnsw.scan_mem_multiplier", smm)
                    set_guc(cur, "hnsw.max_scan_tuples", mst)
                    set_guc(cur, "hnsw.ef_search", ef)
                    assert_plan(cur, sel, queries[0], "items_embedding_idx",
                                f"mst={mst}/smm={smm}/ef={ef}")

                    for q in queries[:WARMUP]:
                        cur.execute(QUERY, (sel, q, K)).fetchall()

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
                        "max_scan_tuples": mst,
                        "scan_mem_multiplier": smm,
                        "iterative_scan": MODE,
                        "ef_search": ef,
                        "recall_mean": round(float(np.mean(recalls)), 4),
                        "rows_returned_mean": round(float(np.mean(counts)), 2),
                        "p50_ms": round(float(np.percentile(lats, 50)), 2),
                        "p95_ms": round(float(np.percentile(lats, 95)), 2),
                        "n_queries": len(queries),
                    }
                    rows.append(r)
                    print(f"  mst={mst:<9d} smm={smm} ef={ef:<5d} recall={r['recall_mean']:.3f} "
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
    p.add_argument("--out", default="results_mst.csv")
    ints = lambda s: [int(x) for x in s.split(",")]
    p.add_argument("--sel", type=ints, default=SELECTIVITIES)
    p.add_argument("--mst", type=ints, default=MSTS)
    p.add_argument("--smm", type=ints, default=SMMS)
    p.add_argument("--ef", type=ints, default=EFS)
    a = p.parse_args()
    main(a.dsn, a.out, a.sel, a.mst, a.smm, a.ef)
