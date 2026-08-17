# pgvector filtered-recall lab

Measures what happens to **pgvector recall and latency when you add a `WHERE` clause**
to a vector search — across filter selectivity, `hnsw.ef_search`, and the
`hnsw.iterative_scan` modes added in pgvector 0.8.

## Why this exists

HNSW is an approximate index. When you combine it with a filter, Postgres searches
the index first and applies the predicate to what comes back. If the filter is
selective, few candidates survive and **recall silently collapses** — the query still
returns rows, they're just the wrong ones. No error, no warning.

pgvector 0.8 added iterative index scans to mitigate this. What is not well
documented anywhere public:

- at which selectivity does recall actually fall off?
- how much does `iterative_scan` recover, and what does it cost in p95 latency?
- does raising `ef_search` substitute for it, or not?

This repo answers those with numbers you can reproduce in one command.

## Quickstart

```bash
docker compose up -d
pip install -r requirements.txt
python load.py            # downloads SIFT-128 (~500MB), loads 100k vectors, builds HNSW
python bench.py           # writes results.csv
python plot.py            # writes recall.png
```

Override the connection with `--dsn`, the row count with `--rows`.

**The container publishes on host port 5433, not 5432.** If you already run PostgreSQL
natively it owns 5432, and connecting to the wrong server surfaces as
`password authentication failed for user "postgres"` — which looks like a credentials
bug and isn't one. 5433 sidesteps it without touching your local install.

## The experiment

| axis | values |
|---|---|
| filter selectivity | 100%, 50%, 10%, 1%, 0.1% |
| `hnsw.iterative_scan` | `off`, `relaxed_order`, `strict_order` |
| `hnsw.ef_search` | 40, 100, 400 |

45 cells, 200 held-out query vectors each, `LIMIT 10`.

**Ground truth** comes from the same database with `enable_indexscan`/`enable_bitmapscan`
turned off — an exact scan over the filtered subset. Same engine, same data, no separate
ground-truth pipeline to disagree with.

`recall = |index_result ∩ exact_result| / |exact_result|`

The filter column is `bucket`, uniform over `0..999`, so `WHERE bucket < N` gives exactly
`N/1000` selectivity. One column, any selectivity, no reload.

## Dataset

SIFT-128-euclidean from ann-benchmarks. **Do not substitute random vectors.** In 128
dimensions, uniform random points have near-identical pairwise distances (concentration
of measure), so ANN recall on them is meaningless and does not resemble real embeddings.

## Known limitations

Stated up front, because a benchmark that hides these is not worth reading.

- **Synthetic filter attribute.** `bucket` is uniform and independent of vector position.
  Real filters (tenant, category, date) correlate with the embedding, which changes
  the shape of the collapse. Correlated-attribute runs are the obvious follow-up.
- **Client-side timing.** Latency includes driver and loopback round-trip. Consistent
  across cells, so comparisons hold; absolute numbers are not server-side timings.
- **Single node, laptop-class hardware.** Relative behaviour is the finding, not throughput.
- **100k rows by default.** Recall cliffs move with dataset size — `--rows` scales it up.
- 10 warmup queries per cell, then one timed run per query. Good enough for p50/p95 across
  200 queries; not a substitute for a sustained load test.

## Tuning notes

`load.py` sets `maintenance_work_mem = 512MB` for the index build. HNSW build time is
extremely sensitive to this — if the graph doesn't fit, pgvector falls back to a much
slower on-disk build. Raise it if your container has the memory.

If you raise it, raise `shm_size` in `docker-compose.yml` to match. Docker defaults
`/dev/shm` to 64MB, and a parallel HNSW build allocates the graph in dynamic shared
memory, so the build fails with:

```
could not resize shared memory segment ... No space left on device
```

That reads like a full disk. It isn't — the host has plenty of room. `shm_size` must
exceed `maintenance_work_mem`.

## Requirements

pgvector **0.8.0 or newer** — `hnsw.iterative_scan` does not exist before that.
`bench.py` checks and exits with a clear message.
