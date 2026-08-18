#!/usr/bin/env python3
"""Measure what scan_mem_multiplier costs in memory once queries run concurrently.

The recall recipe (max_scan_tuples raised, scan_mem_multiplier=4) is measured
single-client everywhere else in this repo. scan_mem_multiplier scales the memory
budget of *each* iterative scan, so the interesting question is what happens when
many scans are in flight at once — a single-client benchmark cannot see it.

Memory is read from the container's cgroup, using the `anon` field of memory.stat
rather than memory.current. memory.current includes page cache, which grows on its
own as the scan touches more of the index and would show up as backend memory that
is not there. shared_buffers is accounted separately as `shmem`, so `anon` is close
to the sum of private backend memory.
"""
import argparse
import csv
import statistics
import subprocess
import sys
import threading
import time

from bench import DATA, K, QUERY, assert_plan, set_guc, to_literal, warm_cache

SEL = 1            # 0.1% — the only place the caps bind
MST = 100_000      # gate, held above the binding point throughout
SMMS = [1, 2, 4]
CONCURRENCIES = [1, 8, 32]
MODE = "relaxed_order"
SECONDS = 15
NQ = 100


def cgroup_anon(container):
    """Private (non-cache, non-shared) memory of everything in the container, bytes.

    Returns None if the field is unavailable, so a missing cgroup degrades the run
    to latency-only rather than reporting a fabricated zero.
    """
    out = subprocess.run(
        ["docker", "compose", "exec", "-T", container, "cat", "/sys/fs/cgroup/memory.stat"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("anon "):
            return int(line.split()[1])
    return None


def worker(dsn, queries, smm, stop, lats, errors):
    import psycopg
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("LOAD 'vector'")
            for name, val in (("hnsw.iterative_scan", MODE),
                              ("hnsw.max_scan_tuples", MST),
                              ("hnsw.scan_mem_multiplier", smm),
                              ("enable_seqscan", "off")):
                cur.execute("SELECT set_config(%s, %s, false)", (name, str(val)))
            i = 0
            while not stop.is_set():
                q = queries[i % len(queries)]
                i += 1
                t = time.perf_counter()
                cur.execute(QUERY, (SEL, q, K)).fetchall()
                lats.append((time.perf_counter() - t) * 1000)
    except Exception as e:                      # a backend dying is a result, not a crash
        errors.append(f"{type(e).__name__}: {e}")


def prepare(dsn, queries, smm):
    """Guard the plan and warm the cache before the cell is timed.

    Two separate hazards. The btree indexes bench_btree.py leaves on the filter columns
    let the planner take btree+sort at this selectivity, which would quietly turn a
    concurrency measurement of HNSW into one of a bitmap scan — hence assert_plan, the
    same guard every other script here uses. And with no warmup the first cell of the
    grid reads cold, so its throughput is a fact about running first rather than about
    scan_mem_multiplier.

    Runs on its own connection that closes before the caller takes the memory baseline,
    so none of this warmup's backend memory is charged to the cell.
    """
    import psycopg
    with psycopg.connect(dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("LOAD 'vector'")
        for name, val in (("hnsw.iterative_scan", MODE), ("hnsw.max_scan_tuples", MST),
                          ("hnsw.scan_mem_multiplier", smm), ("enable_seqscan", "off")):
            set_guc(cur, name, val)
        assert_plan(cur, SEL, queries[0], "items_embedding_idx", f"conc/smm={smm}")
        warm_cache(cur, QUERY, SEL, queries)


def sampler(container, stop, peaks):
    while not stop.is_set():
        v = cgroup_anon(container)
        if v is not None:
            peaks.append(v)
        time.sleep(0.5)


def main(dsn, out, container):
    import h5py

    with h5py.File(DATA) as f:
        queries = [to_literal(v) for v in f["test"][:NQ]]

    if cgroup_anon(container) is None:
        print("WARNING: cgroup memory.stat unreadable — memory columns will be empty",
              file=sys.stderr)

    rows = []
    for smm in SMMS:
        for conc in CONCURRENCIES:
            prepare(dsn, queries, smm)

            # Let the previous cell's backends, and prepare's, exit before baselining,
            # otherwise their memory is attributed to this cell.
            time.sleep(3)
            base = cgroup_anon(container)

            stop = threading.Event()
            lats, errors, peaks = [], [], []
            threads = [threading.Thread(target=worker,
                                        args=(dsn, queries, smm, stop, lats, errors))
                       for _ in range(conc)]
            samp = threading.Thread(target=sampler, args=(container, stop, peaks))
            samp.start()
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            time.sleep(SECONDS)
            stop.set()
            for t in threads:
                t.join()
            samp.join()
            elapsed = time.perf_counter() - t0

            peak = max(peaks) if peaks else None
            r = {
                "scan_mem_multiplier": smm,
                "concurrency": conc,
                "queries": len(lats),
                "qps": round(len(lats) / elapsed, 1),
                "p50_ms": round(statistics.median(lats), 2) if lats else None,
                "p95_ms": (round(statistics.quantiles(lats, n=20)[18], 2)
                           if len(lats) > 20 else None),
                "anon_base_mb": round(base / 2**20, 1) if base else None,
                "anon_peak_mb": round(peak / 2**20, 1) if peak else None,
                "anon_delta_mb": round((peak - base) / 2**20, 1) if peak and base else None,
                "errors": len(errors),
            }
            rows.append(r)
            print(f"  smm={smm} conc={conc:<3d} qps={r['qps']:7.1f} "
                  f"p50={r['p50_ms']:8.2f}ms p95={str(r['p95_ms']):>9}ms "
                  f"anon +{r['anon_delta_mb']}MB (peak {r['anon_peak_mb']}MB) "
                  f"errors={r['errors']}")
            if errors:
                print(f"    first error: {errors[0]}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--out", default="results_conc.csv")
    p.add_argument("--container", default="db")
    a = p.parse_args()
    main(a.dsn, a.out, a.container)
