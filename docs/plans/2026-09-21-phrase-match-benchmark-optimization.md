# Phrase Match Benchmark And Optimization Implementation Plan

**Goal:** Reproduce the scalar-benchmark phrase-match workload in Rust, measure 2/3/5/10/30-term queries at slop 0/1/2 and exact 10%/30%/50%/90% hit rates, and optimize Tantivy's no-scoring phrase-existence path without changing query semantics.

**Architecture:** Use a standalone Cargo benchmark program under `benchmarks/phrase-match/` that path-depends only on this Tantivy checkout and builds a deterministic one-million-document in-memory index from the YAML workload parameters. Inject four disjoint 30-token phrases into exact, nested random samples containing 10%/30%/50%/90% of the documents, then query 2/3/5/10/30-token prefixes so phrase length, slop, and hit rate can be varied independently on one index. Optimize the two-term slop path and multi-term span transition, then compare the same binary before and after the source change.

**Tech stack:** Rust, Tantivy, Criterion dev dependencies only where already available, `std::time::Instant`, seeded `rand`.

---

### Task 1: Add the reproducible benchmark

**Files:**
- Create: `benchmarks/phrase-match/Cargo.toml`
- Create: `benchmarks/phrase-match/src/main.rs`

**Steps:**
1. Generate 1,000,000 documents with seed 42 and 3-12 uniformly distributed filler tokens.
2. Generate one deterministic random document ordering and use prefixes of that ordering to inject four disjoint 30-token phrases into exactly 10%, 30%, 50%, and 90% of rows.
3. Insert each selected phrase as an intact block at a filler-token boundary so later phrase insertions cannot split an earlier phrase.
4. Build one in-memory text index with positions enabled and merge it to a stable segment layout.
5. Build `PhraseQuery` values for 2, 3, 5, 10, and 30 terms with slop 0, 1, and 2 for every hit rate.
6. Assert that every query returns `num_docs * hit_rate / 100` documents exactly.
7. Run 10 warmup iterations and 50 measured iterations per case and print hit rate, median, p95, min, and max latency.
8. Support environment overrides for document, warmup, and iteration counts so smoke validation is fast while the default remains faithful to the YAML.

### Task 2: Capture the baseline

**Files:**
- Do not modify source

**Steps:**
1. Compile the benchmark in release/bench mode.
2. Run the full 1M-document, 10-warmup, 50-iteration matrix.
3. Save the raw output outside the repository for before/after comparison.

### Task 3: Optimize phrase existence

**Files:**
- Modify: `src/query/phrase_query/phrase_scorer.rs`

**Steps:**
1. Add a two-pointer `intersection_exists_with_slop` helper.
2. Route two-term slop existence checks through that helper instead of materializing `PositionSpan` values.
3. Allocate span buffers only for slop queries with more than two terms.
4. Stop scanning `next_positions` as soon as a position inside the current span is found, because no later position can produce a smaller span.

### Task 4: Validate correctness and performance

**Files:**
- Modify tests: `src/query/phrase_query/phrase_scorer.rs`

**Steps:**
1. Add focused helper tests covering no match, boundary slop, reversed positions, and repeated positions.
2. Run the phrase-query unit tests.
3. Run the benchmark smoke configuration.
4. Run the full benchmark with the same 1M/10/50 settings.
5. Compare per-case latency and report regressions as well as improvements.

### Task 5: Expand the selectivity matrix

**Files:**
- Modify: `benchmarks/phrase-match/src/main.rs`
- Modify: `benchmarks/phrase-match/README.md`
- Modify: `benchmarks/phrase-match/results-2026-09-21.md`

**Steps:**
1. Add deterministic exact-hit document ranking and four independent 30-token phrase families.
2. Extend the query lengths with a 30-token case and include hit rate in each CSV row and case name.
3. Add unit tests for the document ranking and uniqueness of phrase tokens.
4. Run crate unit tests, the smoke matrix, formatting checks, and the full 1M/10/50 matrix.
5. Run the 60-case matrix against both the unmodified base commit and the optimized working tree under the same release settings.
6. Record baseline median/p95, optimized median/p95, and median speedup for every case, and call out the exact hit counts for each selectivity.
