#!/usr/bin/env python3
"""Find the exact selectivity where the planner abandons the exact plan for HNSW.

`bucket < N` for N in 1..10 steps selectivity from 0.1% to 1.0% in 0.1% increments,
which is the resolution the 1000-bucket attribute allows.

The plan choice does not depend on the query vector — HNSW costs the same whatever is
being searched for — so the crossover is a property of (column, selectivity) alone and
comes straight from EXPLAIN. That makes it immune to host load, unlike any timing.

Rows returned is measured alongside it for the `far` arm, where the failure lives. It
needs no ground truth: an exact scan of a non-empty subset always returns 10, so any
mean below 10 is the index losing rows outright.
"""
import argparse
import csv

from bench import DATA, K
from bench_corr import COLUMNS, LAYOUTS, anchor_sets, arms_for, query_for
from bench_btree import RECIPE, plan_label

MAX_SEL = 10  # 0.1% .. 1.0% in 0.1% steps
NQ = 10       # only used to confirm the failure, not to time anything


def main(dsn, out, max_sel):
    import h5py
    import numpy as np
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")
        n_rows = cur.execute("SELECT count(*) FROM items").fetchone()[0]
        for name, val in RECIPE.items():
            cur.execute("SELECT set_config(%s, %s, false)", (name, str(val)))

        with h5py.File(DATA) as f:
            test = f["test"][:]
            anchors = anchor_sets(f, n_rows)

        rows = []
        for col in COLUMNS:
            q_sql = query_for(col)
            far = arms_for(col, test, anchors, NQ)["far"]
            for sel in range(1, max_sel + 1):
                plan = "\n".join(r[0] for r in cur.execute(
                    "EXPLAIN " + q_sql, (sel, far[0], K)).fetchall())
                chosen = plan_label(plan)
                est = cur.execute(
                    "SELECT count(*) FROM items WHERE " + col + " < %s", (sel,)
                ).fetchone()[0]

                counts = [len(cur.execute(q_sql, (sel, q, K)).fetchall()) for q in far]
                r = {
                    "layout": LAYOUTS[col],
                    "selectivity_pct": sel / 10.0,
                    "subset_rows": est,
                    "plan": chosen,
                    "far_rows_returned_mean": round(float(np.mean(counts)), 2),
                    "n_queries": len(counts),
                }
                rows.append(r)
                flag = "  <-- returns nothing" if r["far_rows_returned_mean"] == 0 else ""
                print(f"  {r['layout']:12s} {r['selectivity_pct']:4.1f}% "
                      f"({est:6d} rows) {chosen:12s} "
                      f"far rows={r['far_rows_returned_mean']:5.2f}{flag}")
            print()

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results_crossover.csv")
    # Raise past 10 when the filter column is physically clustered: the exact plan then
    # stays cheapest well beyond 1% and the flip falls outside the default range.
    p.add_argument("--max-sel", type=int, default=MAX_SEL)
    a = p.parse_args()
    main(a.dsn, a.out, a.max_sel)
