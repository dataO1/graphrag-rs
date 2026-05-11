# graphrag-rs eval — LLM-judge harness

Reusable eval harness for A/B comparison of graphrag-rs retrieval modes.
Built for Phase 8 (HippoRAG PPR `mode: deep` vs `mode: hybrid` baseline);
inherited by Phase 9 (BM25 fusion) and Phase 10 (query decomposition).

## Files

| File | Purpose |
|---|---|
| `hipporag-vs-baseline.csv` | Gold-set (≥20 hand-curated multi-hop prompts) |
| `run-eval.sh` | Runner: POSTs each prompt per mode, outputs JSONL |
| `judge.sh` | Judge driver: LLM-scores each JSONL record, writes results MD |
| `raw-<DATE>.jsonl` | Runner output (gitignored) |
| `judged-<DATE>.jsonl` | Judge output (gitignored) |
| `results-<DATE>.md` | Human-readable report (committed after each eval run) |

## Quick start

```bash
# 1. Run against a live graphrag-server (must be running at 127.0.0.1:17180)
./eval/run-eval.sh

# 2. Judge with the LLM router (Spark at 127.0.0.1:17170)
./eval/judge.sh

# 3. Review
cat eval/results-$(date +%Y-%m-%d).md
```

## Smoke-test (before card 3 deploys hipporag mode)

```bash
# Only run the hybrid (=default) mode; hipporag mode returns 422 until card 3
./eval/run-eval.sh --modes hybrid
./eval/judge.sh
```

## Adding a new mode (Phase 9, Phase 10, …)

1. Add the new server-side `mode` value to `QueryMode` enum in the server.
2. Map the user-facing token in `run-eval.sh`'s `map_mode()` function if needed.
3. Run: `./eval/run-eval.sh --modes hybrid,hipporag,bm25_fusion`
4. The judge automatically handles any number of mode columns.

## Gold-set shape coverage

The 21 prompts in `hipporag-vs-baseline.csv` cover five entity shapes from
the user's vault:

| Shape | Prompts | Entity / topic |
|---|---|---|
| WCDC | 1–5 | WCDC DataCross compliance platform, API, bugs, roles |
| Yageo | 6–7 | Yageo scraper project, yageogroup.com, Quarkus |
| tasks | 8–10 | Work + personal task tracking, journal queries |
| Christian (SEMLA people) | 11–13 | SEMLA co-authors (Jan, Jochen, Valentin, Daniel), ISMS crypto/VPN |
| oVirt / SEMLA infra | 14–15 | SEMLA VM setup, AlmaLinux, QCOW2, NFS, migration plans |
| graphrag-rs | 16–21 | HippoRAG algorithm, OOM fix, Spark setup, LLM router |

> Note: "Christian" shape targets multi-hop person+project queries anchored
> in the SEMLA ISMS notes (the same recall shape as "who does X in project Y?").
> "oVirt" shape targets the SEMLA virtualisation infrastructure notes where
> VM/hypervisor topics appear.

## CSV format

```
prompt | expected_answer_summary | must_recall_substrings
```

`must_recall_substrings`: `|`-separated list; any one substring matching
(case-insensitive) in the retrieved chunks counts as a context hit.

## Judge model

Default: `local-qwen3.6` via `127.0.0.1:17170` (llm-router nginx).
- Primary: Spark 1 (`172.16.51.90:8000`, Qwen3.6-27B-NVFP4)
- Fallback: local llama-server-qwen36 at 17171

Override: `./eval/judge.sh --model sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`

## Acceptance criterion (card 6)

Judge-aggregate delta ≥ +10% for `mode: hipporag` vs `mode: hybrid` on
the gold set, AND no regression on direct-lookup baseline questions
(prompts 16–21, which test well-indexed factual recall).

## Scoring rubric

| Score | Meaning |
|---|---|
| 5 | Fully correct / directly answers / perfectly coherent |
| 4 | Mostly correct with minor gaps |
| 3 | Partially correct, key points present |
| 2 | Mostly wrong but not entirely off-topic |
| 1 | Entirely wrong, off-topic, or empty |
