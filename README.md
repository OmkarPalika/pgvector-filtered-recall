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

## Results

Run at 100k and 1M SIFT-128 vectors, pgvector 0.8.6 on PostgreSQL 17. Defaults
(`ef_search=40`, `iterative_scan=off`), asking for `LIMIT 10`:

| selectivity | recall@10 100k | recall@10 1M | rows returned 1M |
|---|---|---|---|
| 100% | 0.980 | 0.945 | 10.00 |
| 10% | 0.398 | 0.406 | 4.06 |
| 1% | 0.046 | 0.041 | 0.41 |
| 0.1% | 0.004 | 0.006 | 0.06 |

**The cliff is set by selectivity, not by table size.** Ten times the data moves
the recall at a given selectivity barely at all. What does change with scale:

- **The mitigation weakens.** `iterative_scan=relaxed_order` fully recovers recall at
  100k (0.975 at 0.1%) but only reaches 0.848 at 1M, and does not respond to
  `ef_search` — 0.848 at ef=40, 0.866 at ef=400, both around 100ms.
- **The planner's accidental rescue disappears.** At 100k and 0.1% selectivity the
  planner drops HNSW for an exact seq scan, so results are correct by accident. At 1M
  a seq scan over 1.36GB is no longer cheap, so it stays on HNSW — and the stock
  configuration returns 0.06 of the 10 rows requested, with no error.

So the small-scale run understates the problem twice over: the fix looks complete when
it isn't, and the planner covers the worst case when it won't at production size.

### The two caps have to be raised together

`sweep_mst.py` varies the two knobs left once `ef_search` stops responding. At 1M rows,
0.1% selectivity, `relaxed_order`, recall@10:

| | `max_scan_tuples=20000` (default) | `max_scan_tuples>=50000` |
|---|---|---|
| `scan_mem_multiplier=1` (default) | 0.859 | 0.868 |
| `scan_mem_multiplier=2` | 0.859 | **0.973** |

Neither knob does anything on its own. `max_scan_tuples` alone saturates at 0.868 and
stays there all the way to 1,000,000. `scan_mem_multiplier` alone changes nothing at all.

An iterative scan stops at whichever limit it reaches first. The stock configuration
stops on the tuple cap, so added memory is never reached; lift only the tuple cap and it
stops on memory instead, which is why 0.868 holds flat across a 20x range of tuple
budget. Only raising both lets the scan run to where the recall is.

Tuning them one at a time — the ordinary way to tune anything — makes both look useless.

### ef_search is the wrong knob

With `max_scan_tuples` raised out of the way, sweeping `ef_search` over a 25x range
changes almost nothing, at every memory setting:

| `scan_mem_multiplier` | ef=40 | ef=1000 | p50 |
|---|---|---|---|
| 1 | 0.868 | 0.886 | ~98ms |
| 2 | 0.973 | 0.977 | ~161ms |
| 4 | **0.998** | 0.999 | ~217ms |

25x more `ef_search` buys 0.018 recall. One step of `scan_mem_multiplier` buys 0.105.

The three settings do different jobs, and only one of them is a tuning dial:

- **`max_scan_tuples` is a gate.** Below the binding point nothing else can take effect;
  above it, further increases do nothing whatsoever.
- **`scan_mem_multiplier` is the dial.** 1 → 2 → 4 gives 0.868 → 0.973 → 0.998 at
  98 → 161 → 217ms — a clean recall-for-latency trade.
- **`ef_search` is inert under a selective filter.** Not weak, inert. It is also the
  setting almost every tuning guide reaches for first.

The ceiling is not structural. 0.998 recall at 0.1% selectivity on 1M rows is reachable
for about 2.2x the default latency. The default configuration is simply the wrong one
for filtered search.

At 1% selectivity every cell is 0.993 and neither cap binds. All of this appears only
below roughly 1%.

### What the recipe costs under concurrency

`scan_mem_multiplier` scales the memory budget of *each* iterative scan, so the cost is
paid per in-flight query. `bench_conc.py` measures it with the container's cgroup v2
`anon` counter — not `memory.current`, which includes page cache and shared_buffers and
would report growth that is not backend memory.

Private backend memory above idle, 0.1% selectivity, `max_scan_tuples=100000`:

| `scan_mem_multiplier` | conc=1 | conc=8 | conc=32 | per connection |
|---|---|---|---|---|
| 1 | 5.8MB | 46.1MB | 179.5MB | ~5.7MB |
| 2 | 9.8MB | 78.2MB | 302.9MB | ~9.6MB |
| 4 | 15.4MB | 128.9MB | 532.9MB | **~16.4MB** |

Linear in both axes — no superlinearity and no errors, which makes it budgetable:
**allow roughly 16MB per concurrent filtered query at `smm=4`.** A 100-connection pool
wants ~1.6GB of headroom beyond `shared_buffers` for scan memory alone.

Throughput pays too. At 32 concurrent clients, going from `smm=1` to `smm=4` drops
throughput from 124 to 48 qps and raises p95 from 367ms to 1004ms. The single-client
figure of 2.2x latency understates it; under load it is about 2.6x throughput as well.

So the recipe is conditional, not universal. 0.998 recall for ~3x memory and ~2.6x
throughput is an easy trade for a low-qps internal search and a poor one for a
high-qps user-facing endpoint that has not been capacity-planned for it.

Caveat: the container ran with no memory limit, so this measures the cost, not the
point at which a constrained instance would fail.

### A uniform filter attribute describes neither real case

Everything above filters on `bucket`, which is uniform random and independent of vector
position — the standard way filtered-ANN benchmarks are built. Real filter attributes
are not independent: one tenant's documents, one category, one date range sit together
in embedding space. `bucket_corr` models that, ranking rows by distance to a fixed
anchor so `bucket_corr < N` is a compact ball holding exactly the same N/1000 of the
table. Mean distance to the anchor is 253 for the correlated subset against 506 for the
uniform one.

Where the query sits relative to that ball decides the outcome. Both cases, 1M rows,
`recall@10` and rows returned out of 10 requested:

| layout | query | selectivity | default | recipe |
|---|---|---|---|---|
| uniform | near | 0.1% | 0.003 (0.03 rows) | 1.000, 183ms |
| uniform | far | 0.1% | 0.002 (0.02 rows) | 0.999, 160ms |
| correlated | near | 0.1% | **0.471** (4.82 rows) | 0.945, 7ms |
| correlated | near | 1.0% | **0.913** (9.83 rows) | 0.922, 4ms |
| correlated | far | 0.1% | **0.000** (0 rows) | **0.000, 702ms** |
| correlated | far | 1.0% | **0.000** (0 rows) | **0.000, 765ms** |

Three things follow, and they cut in opposite directions:

- **Searching inside your own region is far better than a uniform benchmark suggests.**
  0.471 against 0.003 on stock settings. The filtered rows are where the graph walk
  already goes, so candidates survive the predicate instead of being discarded.
- **The fix is much cheaper there too** — 0.471 to 0.945 for 3.7ms to 7.2ms, against
  183ms to do the same job on uniform data.
- **Searching outside the region fails completely, and the fix makes it worse.** Zero
  rows at every setting, and `relaxed_order` spends 700ms+ to return nothing where the
  default fails in 3.5ms. Confirmed by EXPLAIN: `Rows Removed by Filter: 101373`, the
  entire `max_scan_tuples` budget spent, no row surviving the predicate.

That last row is a query for something the tenant does not have, which is ordinary. It
means **the queries that return nothing are the most expensive queries in the system**,
and combined with ~16MB of scan memory each, a burst of no-match searches is the worst
load this configuration can be given: peak memory, peak latency, no results.

So a uniform filter attribute is not a conservative simplification. It sits between two
real behaviours and describes neither.

## Dataset

SIFT-128-euclidean from ann-benchmarks. **Do not substitute random vectors.** In 128
dimensions, uniform random points have near-identical pairwise distances (concentration
of measure), so ANN recall on them is meaningless and does not resemble real embeddings.

## Known limitations

Stated up front, because a benchmark that hides these is not worth reading.

- **Two filter attributes, both synthetic.** `bucket` is uniform and independent of
  vector position; `bucket_corr` is a compact ball around a fixed anchor. Real
  attributes sit somewhere between a single ball and many scattered clusters, and a
  multi-cluster attribute is the obvious next one to add.
- **Client-side timing.** Latency includes driver and loopback round-trip. Consistent
  across cells, so comparisons hold; absolute numbers are not server-side timings.
- **Single node, laptop-class hardware.** Relative behaviour is the finding, not throughput.
- **100k rows by default.** `--rows` scales it up; results at 1M are in `results_1m.csv`.
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
