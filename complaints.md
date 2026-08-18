# Filtered vector search — evidence log

Real people hitting recall/filter problems with pgvector. Target: **30+ distinct people**
from the last 12 months. Under 10 means the pain isn't there and the thesis dies cheap.

Sources swept: `pgvector/pgvector` issues (via `gh search issues`), `run-llama/llama_index`,
`timescale/pgvectorscale`, `supabase/supabase`, plus vendor and practitioner writeups.
Reddit and Stack Overflow are not reachable by the search tool used here and remain
un-swept — see [Gaps](#gaps).

**Result: 39 distinct people across 40 threads. 11 inside the 12-month window, the rest before it.** The headline
number clears the bar; the date distribution does not, and that is the finding worth
reading. See [What the dates say](#what-the-dates-say).

Sorted newest first. `pgv#N` is `github.com/pgvector/pgvector/issues/N`.

| # | date | source | link | the complaint, in their words |
|---|------|--------|------|-------------------------------|
| 1 | 2026-07-28 | llama_index | [llama#22475](https://github.com/run-llama/llama_index/issues/22475) @microbluey | "PGVectorStore interpolates metadata filter keys into SQL" |
| 2 | 2026-07-24 | pgvector | [pgv#1001](https://github.com/pgvector/pgvector/issues/1001) @varadfromeast | Built a hypothetical-index prototype because "pgvector's cost function reads lists from the physical index" — you cannot test whether the planner will pick the vector index without building it |
| 3 | 2026-04-29 | pgvector | [pgv#980](https://github.com/pgvector/pgvector/issues/980) @anuj-kukunarapu | "instead of traversing a single global vector graph and filtering results afterward" — asks for filtered HNSW / segment-level indexes for `tenant_id`, `category_id` filters |
| 4 | 2026-01-21 | pgvector | [pgv#949](https://github.com/pgvector/pgvector/issues/949) @brycechesternewman | Partitioning 250M rows into 10M-row chunks purely to keep HNSW builds inside `maintenance_work_mem`; unsure how to set parameters per partition |
| 5 | 2025-11-10 | pgvector | [pgv#923](https://github.com/pgvector/pgvector/issues/923) @vrana | **"Without the index, my query returns ~13k rows. With the index, it returns 4 rows (not 4k, really only 4)."** Adds: "Sometimes, when I drop the index and recreate it then the problem is fixed. But sometimes it's not." |
| 6 | 2025-10-26 | pgvector | [pgv#916](https://github.com/pgvector/pgvector/issues/916) @igm503 | 3M vectors, date filter. `LIMIT 5` takes 1,885ms; `LIMIT 40` on the identical filter takes 235ms. "I'm curious about why postgresql with pgvector makes the decisions it does" |
| 7 | 2025-09-26 | pgvector | [pgv#902](https://github.com/pgvector/pgvector/issues/902) @gustavfridell | Filter by date first to cut millions of rows to a few thousand, then scan — "In practice, though, this approach is too slow for real-world usage." |
| 8 | 2025-09-26 | pgvector | [pgv#901](https://github.com/pgvector/pgvector/issues/901) @Wulfsta | Extending pgvector with a custom distance metric rather than work within it |
| 9 | 2025-08-25 | pgvector | [pgv#891](https://github.com/pgvector/pgvector/issues/891) @brycechesternewman | 500M rows: "I know this question is dependent what other data might be used as filters." Asking whether to partition *because of* the filter interaction |
| 10 | 2025-07-23 | pgvector | [pgv#878](https://github.com/pgvector/pgvector/issues/878) @JoranDox | Partial HNSW index over "somewhere between 0 and 10%" of the table "gives partial results as if filtering" |
| 11 | 2025-06-24 | MongoDB / dev.to | [Franck Pachot](https://dev.to/franckpachot/no-pre-filtering-in-pgvector-means-reduced-ann-recall-1aa1) | "Post-filtering reduces recall even further, as some candidates are discarded, leading to the possibility of missing good matches." Asked for 15, got 11: 21 of 40 candidates discarded by the metadata filter |
| 12 | 2025-06-12 | llama_index | [llama#19060](https://github.com/run-llama/llama_index/issues/19060) @dawancha | "Forced `::float` Type Cast in pgvector Metadata Filter Disables Index Usage" |
| 13 | 2025-06-04 | pgvector | [pgv#850](https://github.com/pgvector/pgvector/issues/850) @bennomeyer | "the query returns incorrect results that don't respect the distance-based ordering" when LIMIT meets ORDER BY on distance |
| 14 | 2025-05-25 | pgvector | [pgv#845](https://github.com/pgvector/pgvector/issues/845) @developerayuva | Partitioned table, ~100k rows: "This returned 160 rows (40 from each partition), which matches the default ef_search = 40. But when I reduce the LIMIT to 100:" |
| 15 | 2025-05-22 | pgvector | [pgv#841](https://github.com/pgvector/pgvector/issues/841) @bhagyajitjagdev | **"vector similarity queries return 0 results when using LIMIT values just below a certain threshold, but return expected results when the limit is slightly higher"** — only when combining distance ordering with additional filters |
| 16 | 2025-04-07 | pgvector | [pgv#816](https://github.com/pgvector/pgvector/issues/816) @mahdimanesh | 15M records, 90GB HNSW index on Aurora Serverless V2 with 512GB RAM available — index not used |
| 17 | 2025-02-25 | pgvector | [pgv#785](https://github.com/pgvector/pgvector/issues/785) @shikaiwei1 | **"Adjusting parameters like `hnsw.iterative_scan`, `hnsw.scan_mem_multiplier`, and `hnsw.max_scan_tuples` partially mitigates the issue but introduces significant performance trade-offs."** Filed as a follow-up because pgv#719's fix did not hold |
| 18 | 2025-02-13 | pgvector | [pgv#776](https://github.com/pgvector/pgvector/issues/776) @natehaze | "After a bunch of research, experimentation - and no clear guidance in Postgres documentation" — concludes iterative scan cannot apply to a subquery filter |
| 19 | 2025-01-06 | pgvector | [pgv#751](https://github.com/pgvector/pgvector/issues/751) @svim-ig | **1M records, `ef_search=1000`, `relaxed_order`, `max_scan_tuples=20000`, `scan_mem_multiplier=2` — "Not getting results (getting empty records) with HNSW and filtering"** |
| 20 | 2024-12-07 | pgvector | [pgv#727](https://github.com/pgvector/pgvector/issues/727) @aropb | "Why is the hnsw index not used?" — tags array + jsonb payload filter |
| 21 | 2024-12-02 | pgvector | [pgv#722](https://github.com/pgvector/pgvector/issues/722) @hinthornw | After 0.8.0 shipped iterative scan: "it seems the current index usage is still restricted to equivalence filters(?)" |
| 22 | 2024-11-28 | pgvector | [pgv#721](https://github.com/pgvector/pgvector/issues/721) @rsomani95 | "the query planner chooses not to use the index in two conditions... Values of the filters in the `WHERE` clause return a larger number of values" — btree indexes already present on both filter columns |
| 23 | 2024-11-26 | pgvector | [pgv#719](https://github.com/pgvector/pgvector/issues/719) @prathier | 7M rows: **"If I increase hnsw.ef_search from 100 to 1000 it works but it's slower. And I suppose that 1000 will not be enough when my table will be bigger."** |
| 24 | 2024-10-28 | pgvector | [pgv#703](https://github.com/pgvector/pgvector/issues/703) @kinghuang | 1.6M rows: "the query planner always chooses a sequential scan over using a HNSW index" — ~75 seconds |
| 25 | 2024-09-16 | pgvector | [pgv#675](https://github.com/pgvector/pgvector/issues/675) @rmincling | "Setting ef_search to different values does not affect number of results retrieved" (Django + HNSW) |
| 26 | 2024-09-11 | pgvector | [pgv#671](https://github.com/pgvector/pgvector/issues/671) @vinodhbalasubramanian | **"Postgres gives inconsistent result count when it uses the HNSW index vs index not being used."** |
| 27 | 2024-08-27 | pgvector | [pgv#662](https://github.com/pgvector/pgvector/issues/662) @Jontpan | "the query planner grossly underestimating the performance of index searching over sequential" |
| 28 | 2024-07-25 | pgvectorscale | [scale#116](https://github.com/timescale/pgvectorscale/issues/116) @alanwli | "Poor recall/throughput perf vs. pgvector on small/low-dimension datasets" |
| 29 | 2024-07-22 | pgvector | [pgv#630](https://github.com/pgvector/pgvector/issues/630) @Sankar-A | ~1M embeddings, paginated search: "search query returning inconsistent results based on the limit" — count query at LIMIT 200 disagrees with the page at LIMIT 10 |
| 30 | 2024-07-12 | pgvectorscale | [scale#109](https://github.com/timescale/pgvectorscale/issues/109) @mgrosso | "How can I ensure a streaming filter of diskann results rather than a table scan or other index?" |
| 31 | 2024-05-27 | pgvector | [pgv#575](https://github.com/pgvector/pgvector/issues/575) @gkourie | "there seems to be an unexpected behavior related to filtering" |
| 32 | 2024-05-10 | pgvector | [pgv#553](https://github.com/pgvector/pgvector/issues/553) @zhrt123 | "Lack of result when selecting data without limit" — under 1000 rows the tests expect Seq Scan and get Index Scan |
| 33 | 2024-05-06 | pgvector | [pgv#543](https://github.com/pgvector/pgvector/issues/543) @jpbalarini | "I get different results when my query uses the HNSW index than when it does not." |
| 34 | 2024-03-05 | pgvector | [pgv#480](https://github.com/pgvector/pgvector/issues/480) @Keeo | 14M rows, 90GB table: "Query with HNSW index returns only small portion of results" |
| 35 | 2025-02-19 | llama_index | [llama#17857](https://github.com/run-llama/llama_index/issues/17857) @FadhelHaidar | "Metadata filtering does not filter at all" |
| 36 | 2025-05-07 | llama_index | [llama#18648](https://github.com/run-llama/llama_index/issues/18648) @Raayhaann | "PGVectorStore: Incorrect SQL (float cast) for EQ filter on JSONB metadata field" |
| 37 | 2023-12-14 | llama_index | [llama#9519](https://github.com/run-llama/llama_index/issues/9519) @rendyfebry | "Metadata Filter won't work properly on HNSW PgVectorStore due to PgVector limitation" |
| 38 | 2023-09-12 | pgvector | [pgv#263](https://github.com/pgvector/pgvector/issues/263) @vincenzon | Two identical columns, one indexed: "when I run the same query using the indexed column I get no results" |
| 39 | 2023-09-11 | pgvector | [pgv#259](https://github.com/pgvector/pgvector/issues/259) @Palmik | "I would like to understand how the current implementation handles HNSW + filtering" — the origin thread, still referenced in 2024 issues |
| 40 | 2023-08-29 | pgvector | [pgv#244](https://github.com/pgvector/pgvector/issues/244) @alanwli | "User sets `ef_search` to 10 expecting to get top-10 results back. But the top-10 results in the HNSW index happen to be dead tuples... then the query will return 0 results." |

`alanwli` appears twice (pgv#244 and scale#116); the distinct-people count collapses them.

## What the dates say

40 threads, **39 distinct people** (`alanwli` filed in two repos). Inside the 12-month window (since 2025-08-18): **11**. Before it: the rest.

The raw count clears the bar the template set. The distribution does not, and pretending
otherwise would be the expensive mistake. Complaint volume peaks in 2024 and thins through
2025-2026. Two readings, and they lead to opposite decisions:

**Reading A — it got fixed.** Iterative scan shipped in 0.8.0 (Oct 2024). The acute
"returns zero rows" reports cluster before and just after it. If that closed the wound,
the consulting thesis is built on a solved problem.

**Reading B — it did not get fixed, people stopped reporting.** Three pieces of evidence
against Reading A, all from *after* 0.8.0:

- pgv#751 (Jan 2025) sets `iterative_scan=relaxed_order`, `ef_search=1000`,
  `max_scan_tuples=20000`, `scan_mem_multiplier=2` at 1M records and still gets empty
  results. That is very close to the recipe this repo measured.
- pgv#785 (Feb 2025) is a re-open of pgv#719 stating the new knobs "partially mitigate the
  issue but introduce significant performance trade-offs."
- pgv#923 (Nov 2025) still reports 13k rows becoming 4.

And this repo's own measurement: at 1M rows with a correlated filter, `relaxed_order` plus
raised caps returns **zero rows** in the `far` arm, and `ef_search` is inert.

Reading B is better supported, but **the thinning is real and unexplained**, and "the
maintainer closes these quickly so people stop filing" is a story, not evidence. This is
the single biggest open risk in the thesis and it is not resolved by this sweep.

## What is already commercial

The pain is not undiscovered. Companies publish against it:

- [ParadeDB — "pgvector Limitations"](https://www.paradedb.com/learn/postgresql/pgvector-limitations)
- [ClickHouse — "when to go hybrid"](https://clickhouse.com/resources/engineering/scale-vector-search-postgres)
- [MongoDB (Franck Pachot) — "No pre-filtering in pgvector means reduced ANN recall"](https://dev.to/franckpachot/no-pre-filtering-in-pgvector-means-reduced-ann-recall-1aa1)
- [Crunchy Data — hybrid search patterns](https://www.crunchydata.com/blog/hybrid-vector-search)
- [Curator: Efficient Vector Search with Low-Selectivity Filters](https://arxiv.org/pdf/2601.01291) (academic, Jan 2026)

Cuts both ways. It confirms the pain is worth money to somebody. It also means the
positioning is not "nobody knows about this" — it has to be "nobody has measured it this
precisely." That is a defensible claim: none of the above publishes a selectivity sweep,
a crossover point, or the zero-row failure with reciprocal controls.

## Gaps

- **Reddit and Stack Overflow are un-swept.** The search tool cannot reach either domain.
  These are where non-GitHub-native practitioners complain, so the 9-in-window figure is a
  floor, not a measurement.
- **Discords un-swept** (LangChain, LlamaIndex, Supabase) — not searchable without joining.
- **Nobody here has been contacted.** Every row is a public artifact, not a conversation.
  Willingness to pay is still entirely untested.

## Notes

Distinct *people*, not distinct messages — one person posting five times is one row.
Quote them verbatim. These quotes open the writeup; paraphrase reads as invented.
