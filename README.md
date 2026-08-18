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
| 100% | 0.979 | 0.942 | 10.00 |
| 10% | 0.398 | 0.401 | 4.01 |
| 1% | 0.046 | 0.041 | 0.41 |
| 0.1% | 0.004 | 0.005 | 0.05 |

**The cliff is set by selectivity, not by table size.** Ten times the data moves
the recall at a given selectivity barely at all. What does change with scale:

- **The mitigation weakens.** `iterative_scan=relaxed_order` fully recovers recall at
  100k (0.975 at 0.1%) but only reaches 0.847 at 1M, and does not respond to
  `ef_search` — 0.847 at ef=40, 0.865 at ef=400, 99ms and 106ms.
- **The planner's accidental rescue disappears.** At 100k and 0.1% selectivity the
  planner drops HNSW for an exact seq scan, so results are correct by accident. At 1M
  a seq scan over 1.36GB is no longer cheap, so it stays on HNSW — and the stock
  configuration returns 0.05 of the 10 rows requested, with no error.

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
- **`scan_mem_multiplier` is the dial.** 1 → 2 → 4 gives 0.861 → 0.970 → 0.996 at
  107 → 196 → 237ms — a clean recall-for-latency trade.
- **`ef_search` is inert under a selective filter.** Not weak, inert. It is also the
  setting almost every tuning guide reaches for first.

The ceiling is not structural. 0.996 recall at 0.1% selectivity on 1M rows is reachable
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
| 1 | 5.8MB | 46.0MB | 181.2MB | ~5.7MB |
| 2 | 9.8MB | 78.1MB | 291.0MB | ~9.1MB |
| 4 | 12.8MB | 116.1MB | 400.7MB | **~12.5MB** |

Linear in both axes — no superlinearity and no errors, which makes it budgetable:
**allow roughly 13MB per concurrent filtered query at `smm=4`.** A 100-connection pool
wants ~1.3GB of headroom beyond `shared_buffers` for scan memory alone.

Throughput pays too. At 32 concurrent clients, going from `smm=1` to `smm=4` drops
throughput from 130 to 39 qps and raises p95 from 375ms to 1463ms. The single-client
figure of 2.2x latency understates it; under load it is about 3.3x throughput as well.

So the recipe is conditional, not universal. 0.996 recall for ~2.2x memory and ~3.3x
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

`bucket_multi` goes one step further: it ranks within each of 8 clusters, so the region
is 8 separated balls — a tenant is several topics, not one. Selectivity is unchanged and
the two regions do not overlap at all.

What decides the outcome is where the query sits relative to that region. `recall@10` at
1M rows, rows returned out of the 10 requested in parentheses:

| layout | query | selectivity | default | recipe |
|---|---|---|---|---|
| uniform | near | 0.1% | 0.003 (0.03) | 1.000 |
| multicluster | near | 0.1% | 0.223 (2.24) | 0.996 |
| correlated | near | 0.1% | **0.462** (4.72) | 0.946 |
| uniform | near | 1.0% | 0.051 (0.51) | 0.995 |
| multicluster | near | 1.0% | 0.771 (7.85) | 0.980 |
| correlated | near | 1.0% | **0.913** (9.84) | 0.922 |
| uniform | far | 0.1% | 0.002 (0.02) | 0.999 |
| multicluster | far | 0.1% | **0.000** (0) | 0.685 (9.28) |
| correlated | far | 0.1% | **0.000** (0) | **0.000** (0) |
| uniform | far | 1.0% | 0.028 (0.28) | 0.999 |
| multicluster | far | 1.0% | **0.000** (0) | 0.959 |
| correlated | far | 1.0% | **0.000** (0) | **0.000** (0) |

Reading down the near rows: the more the filter attribute clusters, the better stock
pgvector does, by two orders of magnitude between uniform and single-ball. The filtered
rows are where the graph walk already goes, so candidates survive the predicate instead
of being discarded. A uniform benchmark badly understates ordinary in-tenant search.

Reading down the far rows gives the opposite, and it is the important one:

**On stock settings, a query whose neighbourhood lies outside the filtered region
returns zero rows — not degraded results, zero — while an exact scan returns ten.**

That holds for both realistic layouts, one cluster or eight. Only the uniform attribute
hides it, reporting 0.002, which still reads as merely poor recall rather than total
failure. Confirmed directly by EXPLAIN on a single far query:
`Rows Removed by Filter: 101373` — the whole `max_scan_tuples` budget spent, not one row
surviving the predicate, 0 returned in 633ms against an exact scan's 10.

The recipe rescues this only when the region is scattered: 0.959 and 0.685 for eight
clusters, still 0.000 for one distant ball. Some cluster is always reachable within the
scan budget when there are eight; when there is one and it is far, none is.

A query for something the tenant does not have is ordinary. So the queries that return
nothing are among the most expensive in the system, and at ~16MB of scan memory each a
burst of no-match searches is the worst load this configuration can be given: peak
memory, peak latency, no results.

A uniform filter attribute is therefore not a conservative simplification. It sits
between real behaviours and describes none of them.

### The fix is a btree on the filter column, not HNSW tuning

None of the above is really a vector-index problem. With no index on the filter column,
the planner's only alternative to HNSW is a 1M-row seq scan, which it rightly refuses.
Given a btree it reads the filtered rows directly and sorts them exactly — nothing can
be missed, so there is no recall question at all. It is also why the 100k run never
showed the failure: a seq scan was cheap enough there that the planner already escaped.

The same query that returns 0 rows through HNSW — 2.8ms at stock settings, or 683ms
if the recipe is applied, which buys nothing because the answer is still empty:

```
Limit  (actual time=3.683..3.687 rows=10 loops=1)
  ->  Sort  (Sort Method: top-N heapsort  Memory: 25kB)
        ->  Bitmap Heap Scan on items  (actual rows=1000)
              Recheck Cond: (bucket_corr < 1)
              ->  Bitmap Index Scan on items_bucket_corr_idx  (actual rows=1000)
Execution Time: 3.722 ms
```

10 rows, exact, in 3.7ms — that EXPLAIN was captured while the host was saturated, so
it is an upper bound; the same cell now measures 3.6ms on a warm cache. With a btree present the planner picks `bitmap+sort` at 0.1% for every layout and
both query arms, returning recall 1.000 and a full 10 rows. The zero-row failure is gone.

**An existence probe does not work as a predictor**, which is worth recording because it
is the obvious first idea. `SELECT 1 FROM items WHERE bucket_corr < 1 LIMIT 1` returns a
row in 0.175ms in exactly the failing case: thousands of rows match the predicate, they
are simply nowhere near the query.

The crossover sits between 0.1% and 1%. At 1% the planner switches back to HNSW, and the
failure returns with it:

| layout | arm | selectivity | plan chosen | recall | rows |
|---|---|---|---|---|---|
| correlated | far | 0.1% | bitmap+sort | 1.000 | 10 |
| correlated | far | 1.0% | **hnsw** | **0.000** | **0** |
| correlated | far | 10% | **hnsw** | **0.002** | **0.02** |

At 1% the planner prefers HNSW to a bitmap scan of 10,000 rows and gets nothing back.
(The 0.002 at 10% is one query in 50 returning one row, on a re-measurement against a
different graph; the first run scored a clean 0.000. Either way the plan is useless.)
**The cost model cannot represent this failure**: it prices HNSW as though the index will
find matching rows, and correlation between the filter column and vector position means
it will not. No amount of `ANALYZE` helps, because the statistic that would predict it
does not exist. So the switch is silent, and it happens at a selectivity where the
alternative is still cheap.

### Where exactly the plan flips, and what moves it

Stepping selectivity in 0.1% increments puts the switch between 0.1% and 0.2% for all
three layouts, and the failure appears on precisely the same step:

| layout | 0.1% (1000 rows) | 0.2% (2000 rows) |
|---|---|---|
| uniform | bitmap+sort, 10 rows | hnsw, 10 rows |
| multicluster | bitmap+sort, 10 rows | hnsw, 9.9 rows |
| correlated | bitmap+sort, 10 rows | **hnsw, 0 rows** |

`correlated`/far returns a full 10 rows on the exact plan and nothing at all one step
later, then nothing for every larger selectivity tested. The switch and the failure are
the same event.

That threshold is **not** a fixed row count, and not a percentage of the table either.
Bisecting the same decision on `id`, which is physically ordered, puts it at 9,669 rows
— five to ten times higher than the ~1,000–2,000 seen on the bucket columns:

| filter column | `pg_stats.correlation` | exact plan wins up to |
|---|---|---|
| `id` | 1.00 | ~9,670 rows |
| `bucket_corr` | 0.10 | ~1,000–2,000 rows |
| `bucket_multi` | -0.01 | ~1,000–2,000 rows |
| `bucket` | -0.00 | ~1,000–2,000 rows |

What sets it is heap pages touched, not rows. 9,669 sequential ids occupy a short run of
contiguous pages; 1,000 scattered rows touch about 970, nearly a page each, and the
bitmap heap scan is priced accordingly.

### CLUSTER on the filter column widens the safe band ~8x

That mechanism predicts a lever the pgvector documentation does not mention, so it was
tested directly rather than left as inference. `CLUSTER items USING items_bucket_corr_idx`
physically orders the heap by the filter column. Correlations before and after:

| column | before | after |
|---|---|---|
| `bucket_corr` | 0.10 | **1.00** |
| `id` | 1.00 | **0.07** |
| `bucket` | -0.00 | 0.00 |
| `bucket_multi` | -0.01 | -0.07 |

The two columns swap physical ordering, and their thresholds swap with them:

| filter column | exact plan wins up to (before) | (after) |
|---|---|---|
| `bucket_corr` | ~1,000–2,000 rows | **8,000–9,000 rows** |
| `id` | 9,669 rows | **1,522 rows** |
| `bucket`, `bucket_multi` (controls) | ~1,000–2,000 rows | unchanged |

Nothing changed but physical row order. `bucket_corr` gains about 8x, `id` loses about
6x, and the two untouched columns do not move.

The zero-row failure moves in lockstep with the plan. Clustered, `correlated`/far returns
a full 10 rows from 0.1% through 0.8% and nothing at all from 0.9% on — the boundary that
previously sat at 0.2%. So clustering does not merely shift a cost decision, it widens
the band in which the query is structurally incapable of silently returning nothing.

Two costs belong with that advice. The CLUSTER took **19m48s** on 1M rows under an
ACCESS EXCLUSIVE lock, so the table is unavailable for the duration. And PostgreSQL does
not maintain clustering: it decays as rows are inserted and updated, making this periodic
maintenance rather than a one-time fix.

Practical reading: below roughly 1,000–2,000 matching rows, index the filter column and
let the planner run an exact scan instead of tuning HNSW at all. Physically clustering
that column extends the range about eightfold. Above it, check EXPLAIN for the plan you
actually get, because the choice flips without warning and one side of the flip can
return nothing.

### Latency reproduces across index builds, to within 6%

An earlier version of this section claimed the opposite: that a fresh HNSW graph moves
latency by roughly 2x while recall stays put. That was wrong. `bench_builds.py` is what
disproved it.

It rebuilds only the index, five times, over an untouched table, and re-measures the same
five cells on each build. Ground truth is computed once before the loop, so every build is
scored against identical answers. Each build is measured twice, separated by a full sweep
of the other cells — builds run one after another, so anything drifting with wall-clock
time (host load, thermals, cache state) lands on whichever build was running and is
otherwise indistinguishable from the build itself. That second pass is the control.

p50 per build, median of the two passes:

| cell | b1 | b2 | b3 | b4 | b5 | all | b2-5 |
|---|---|---|---|---|---|---|---|
| uniform near 0.1% recipe | 189.20 | 173.67 | 176.23 | 175.93 | 173.85 | 1.09x | **1.01x** |
| correlated far 1.0% recipe | 835.03 | 694.57 | 700.90 | 696.36 | 705.40 | 1.20x | **1.02x** |
| multicluster far 1.0% recipe | 447.00 | 382.39 | 389.03 | 370.88 | 390.44 | 1.21x | **1.05x** |
| correlated near 0.1% recipe | 8.07 | 7.06 | 6.71 | 6.69 | 6.65 | 1.21x | **1.06x** |
| multicluster near 0.1% default | 4.54 | 3.56 | 3.47 | 3.62 | 3.41 | 1.33x | **1.06x** |

Build 1 is the slowest in all five cells, and its own build took 605s against 462-529s for
the rest — a cold-start cost paid once per process, not a property of its graph. The four
builds after it agree to within 6%, and to within 2% on the two slowest cells. The
within-build control says the same thing from the other side: repeating a cell on the
*same* graph moved it by up to 1.26x, which is more than rebuilding the graph did.

The graphs genuinely do differ. Recall lands on 0.947/0.951/0.952/0.954 across builds for
multicluster/far and 0.944/0.945/0.946/0.948 for correlated/near — impossible if the
parallel build were deterministic. The difference is just far too small to move latency by
anything like 2x.

So the 2x seen across the three full runs above is **not** build nondeterminism, and it is
not host contention either (the idle host was slower). It has not been identified. It
tracks whole-run boundaries — separate process, full reload, hours apart — and none of
those three were isolated from each other.

**Compare latencies within a run.** Still the right rule, but for a weaker reason than the
one given before: not because a rebuilt graph is unreliable, but because whatever moves
cross-run timings has not been pinned down. Recall carries the load-bearing claims in this
README precisely because it reproduces on both axes — ±0.007 across the five builds
here, and ±0.022 worst-case over the full 36-cell btree grid re-measured months of
cache state and one index build apart.

That 36-cell re-measurement is the wider check on all of this, and it is blunt about
which columns are trustworthy:

| | reproduced |
|---|---|
| plan chosen | **36/36 identical** |
| recall | 22/36 moved, worst by 0.022 (uniform/near/10%: 0.950 to 0.928) |
| p50 | 0.5x to 5.4x |

Plan choice is fully deterministic, which is what the crossover claims rest on. Recall
moves in the third decimal. Latency moves by multiples and carries no claim in this
README that a within-run comparison does not already carry.

That re-measurement also exposed a bug in the harness rather than in Postgres. Every
sweep here measures several configurations back to back over the same cell, and each
used a truncated warmup — 5 or 10 queries out of 50 to 200. The first configuration in
the loop therefore absorbed the cold cache and read slower for that reason alone, which
lands in the results looking like an effect of the setting. On uniform/near/1% the two
configurations chose the identical plan and scored the identical recall, and were
recorded 5.3x apart.

The fix is to warm with exactly the queries about to be timed (`warm_cache` in
`bench.py`), so no configuration is privileged by position. A merely larger fixed
warmup shrinks the bias without removing it. Across the twelve cells where both
configurations pick the same plan — where the ratio must be 1.00, since it is the same
query with the same settings — the worst deviation went from **4.37 to 0.12**, and the
median ratio from 1.09 to 0.99.

Re-running the whole grid after the fix, on the same index build, reproduced 36/36
plans and every recall to **0.0000**. So recall does not drift run to run at all; it
moves only when the graph is rebuilt, and then by less than 0.03. The 0.022 above is a
cross-build number, not a noise floor.

`bench_builds.py` keeps a deliberately partial warmup. Its second pass exists to
measure the residual rather than remove it, which is what produced the `within_spread`
column above.

The comparisons that matter are all within-run anyway. From one run,
`correlated`/far:

| selectivity | plan | recall | p50 |
|---|---|---|---|
| 0.1% | bitmap+sort | 1.000 | 3.6ms |
| 1.0% | hnsw | 0.000 | 682ms |

190x slower and returns nothing, same build and same queries. That contrast held in every
run at every magnitude — the ratio moves with cache state, the zero does not.

One anomaly worth recording rather than hiding: `multicluster`/far at 0.1% scores 0.998
on `bitmap+sort`, and an exact plan cannot miss rows. It is one query in 50 losing a
single row to a distance tie broken differently between the two plans — ground truth
sorts under `Seq Scan`, the measurement under `Bitmap Heap Scan`, and `Sort` does not
promise a stable order across plan shapes. Re-measuring the same cell against a
different index build scored a clean 1.000, which is what a tie-break explanation
predicts and a real miss would not. Harmless, but an "exact" plan scoring under 1.0
should be explained, not glossed over.

## Dataset

SIFT-128-euclidean from ann-benchmarks. **Do not substitute random vectors.** In 128
dimensions, uniform random points have near-identical pairwise distances (concentration
of measure), so ANN recall on them is meaningless and does not resemble real embeddings.

## Known limitations

Stated up front, because a benchmark that hides these is not worth reading.

- **Three filter attributes, all synthetic.** `bucket` is uniform and independent of
  vector position, `bucket_corr` is one compact ball, `bucket_multi` is 8 separated
  clusters. Real attributes vary in how tightly they cluster, and the results move a
  long way across that range, so treat the three as bracketing rather than predicting.
- **Build timings vary far more than query results.** The same 1M HNSW build took 469s,
  488s and 1645s on three runs of identical data and parameters, tracking host load.
  Recall figures reproduced across the same runs.
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
