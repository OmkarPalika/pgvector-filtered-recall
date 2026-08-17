#!/usr/bin/env python3
"""Download SIFT-128, load vectors into Postgres, build the HNSW index."""
import argparse
import os
import time
import urllib.request

import h5py
import numpy as np
import psycopg

URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATA = os.path.join("data", "sift-128-euclidean.hdf5")
BUCKETS = 1000
SEED = 42
N_CLUSTERS = 8

# The host 403s the default Python-urllib user agent. Any other UA is accepted,
# so identify the project honestly rather than impersonating a browser.
UA = "pgvector-filtered-recall-lab/0.1"


def progress(blocks, block_size, total, _last=[-1]):
    """Report every 5% only.

    urlretrieve calls this once per 8KB block. Carriage-return overwriting looks
    fine on a TTY but appends to the file when stdout is redirected, so an
    unthrottled hook writes half a megabyte of progress lines into the log.
    """
    if total <= 0:
        return
    pct = int(100 * blocks * block_size / total)
    if pct >= _last[0] + 5:
        _last[0] = pct - pct % 5
        print(f"  {blocks * block_size / 1e6:7.1f} / {total / 1e6:.1f} MB "
              f"({min(pct, 100):3d}%)", flush=True)


def download():
    if os.path.exists(DATA):
        print(f"{DATA} already present")
        return
    os.makedirs("data", exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", UA)]
    urllib.request.install_opener(opener)
    print(f"downloading {URL} (~500MB, one time) ...")
    urllib.request.urlretrieve(URL, DATA + ".tmp", reporthook=progress)
    print()
    os.replace(DATA + ".tmp", DATA)


def to_literal(vec):
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def correlated_buckets(train):
    """Bucket by rank of distance to a fixed anchor.

    `bucket_corr < N` then selects the N/1000 rows closest to the anchor — a compact
    ball in embedding space, which is how real filter attributes behave (one tenant's
    or one category's documents sit together). `bucket` scatters uniformly instead, so
    the pair isolates spatial layout while holding selectivity identical.

    Distances are computed in chunks: train - anchor on the full 1M x 128 array would
    allocate a second 512MB float32 copy, and the squaring another.
    """
    anchor = train[0]  # deterministic, so the attribute is stable across runs
    d = np.empty(len(train), dtype=np.float32)
    for i in range(0, len(train), 100_000):
        chunk = train[i:i + 100_000].astype(np.float32) - anchor
        d[i:i + 100_000] = np.einsum("ij,ij->i", chunk, chunk)

    corr = np.empty(len(train), dtype=np.int32)
    # Equal-sized buckets by rank, so selectivity stays exactly N/1000.
    corr[np.argsort(d)] = np.arange(len(train)) * BUCKETS // len(train)
    return corr


def anchor_indices(n, n_clusters=N_CLUSTERS):
    """Row ids of the cluster anchors. Shared with bench_corr.py so both sides agree
    on where the clusters are without reloading the full training set."""
    return np.sort(np.random.default_rng(SEED).choice(n, n_clusters, replace=False))


def sq_dists(chunk, anchor):
    diff = chunk - anchor
    return np.einsum("ij,ij->i", diff, diff)


def multicluster_buckets(train, n_clusters=N_CLUSTERS):
    """Bucket by rank within the nearest of several anchors.

    One ball (correlated_buckets) is the easy shape: a tenant is really several topic
    clusters scattered across the space. Ranking within each cluster and interleaving
    means `bucket_multi < N` takes the innermost N/1000 of every cluster, so the
    filtered region is n_clusters separated balls at unchanged selectivity.
    """
    anchors = train[anchor_indices(len(train), n_clusters)].astype(np.float32)

    assign = np.empty(len(train), dtype=np.int16)
    dist = np.empty(len(train), dtype=np.float32)
    for i in range(0, len(train), 100_000):
        chunk = train[i:i + 100_000].astype(np.float32)
        # One anchor at a time: chunk[:, None, :] - anchors would allocate
        # 100k x n_clusters x 128 floats.
        d = np.stack([sq_dists(chunk, a) for a in anchors], axis=1)
        assign[i:i + 100_000] = d.argmin(1)
        dist[i:i + 100_000] = d.min(1)

    out = np.empty(len(train), dtype=np.int32)
    for c in range(n_clusters):
        members = np.flatnonzero(assign == c)
        # Rank within the cluster, so each cluster contributes its own N/1000 share.
        out[members[np.argsort(dist[members])]] = (
            np.arange(len(members)) * BUCKETS // len(members))
    return out


def load(dsn, rows):
    with h5py.File(DATA) as f:
        train = f["train"][:rows]
    print(f"loading {len(train)} vectors, dim {train.shape[1]}")

    # Seeded so the filter attribute is identical across runs and machines.
    buckets = np.random.default_rng(SEED).integers(0, BUCKETS, size=len(train))
    corr = correlated_buckets(train)
    multi = multicluster_buckets(train)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            with open("schema.sql") as f:
                cur.execute(f.read())
            conn.commit()

            t = time.perf_counter()
            with cur.copy("COPY items (id, bucket, bucket_corr, bucket_multi, "
                          "embedding) FROM STDIN") as cp:
                for i, v in enumerate(train):
                    cp.write_row((i, int(buckets[i]), int(corr[i]), int(multi[i]),
                                  to_literal(v)))
            conn.commit()
            print(f"copy: {time.perf_counter() - t:.1f}s")

            # HNSW build time is dominated by this. Too small and pgvector falls
            # back to a much slower on-disk build.
            cur.execute("SET maintenance_work_mem = '512MB'")
            t = time.perf_counter()
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
            conn.commit()
            print(f"hnsw build: {time.perf_counter() - t:.1f}s")

            cur.execute("ANALYZE items")
            conn.commit()

            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('items'))")
            print(f"table+index size: {cur.fetchone()[0]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", default="postgresql://postgres:pw@127.0.0.1:5433/vectorlab")
    p.add_argument("--rows", type=int, default=100_000)
    a = p.parse_args()
    download()
    load(a.dsn, a.rows)
