#!/usr/bin/env bash
# eval/judge.sh — LLM-judge driver for graphrag-rs eval harness
#
# Reads raw JSONL from run-eval.sh and POSTs each record to the LLM router
# (Qwen3-Next on Spark via 127.0.0.1:17170) for scoring on three axes:
#   - chunk_relevance  (1–5): did retrieved chunks support the answer?
#   - answer_correctness (1–5): how well does the answer match expected?
#   - coherence        (1–5): does the answer hold together as text?
#
# Judge response: JSON { relevance: int, correctness: int, coherence: int,
#                        reasoning: string }
#
# Output:
#   eval/judged-<DATE>.jsonl  — raw judged records (one per prompt×mode)
#   eval/results-<DATE>.md    — human-readable table + aggregates + Δ
#
# Usage:
#   ./eval/judge.sh [--input RAW.jsonl] [--router URL] [--model MODEL] [--out-dir DIR]
#
# Defaults:
#   --input   eval/raw-<today>.jsonl
#   --router  http://127.0.0.1:17170   (llm-router nginx)
#   --model   local-qwen3.6            (stable alias; routes to Spark 1 primary)
#   --out-dir eval/
#
# Requirements: bash ≥4, curl, jq, python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve python3: prefer PATH, fall back to nix store glob (direnv strips
# PATH in the graphrag-rs dev shell, which has no Python derivation).
if ! command -v python3 &>/dev/null; then
    for _candidate in /nix/store/*python3-3*-env/bin/python3; do
        if [[ -x "${_candidate}" ]]; then
            export PATH="$(dirname "${_candidate}"):${PATH}"
            break
        fi
    done
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
DATE_STAMP="$(date +%Y-%m-%d)"
INPUT_FILE="${SCRIPT_DIR}/raw-${DATE_STAMP}.jsonl"
ROUTER_URL="http://127.0.0.1:17170"
MODEL="local-qwen3.6"
OUT_DIR="${SCRIPT_DIR}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT_FILE="$2"; shift 2 ;;
        --router)  ROUTER_URL="$2"; shift 2 ;;
        --model)   MODEL="$2";      shift 2 ;;
        --out-dir) OUT_DIR="$2";    shift 2 ;;
        --help|-h)
            sed -n '2,/^[^#]/p' "$0" | head -25
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

JUDGED_FILE="${OUT_DIR}/judged-${DATE_STAMP}.jsonl"
RESULTS_FILE="${OUT_DIR}/results-${DATE_STAMP}.md"

# ── Pre-flight ────────────────────────────────────────────────────────────────
for cmd in curl jq python3; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: required command '${cmd}' not found on PATH." >&2
        exit 1
    fi
done

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    echo "Run ./eval/run-eval.sh first to generate raw results." >&2
    exit 1
fi

# ── Judge prompt template ─────────────────────────────────────────────────────
# Stored as a bash function so the caller can inline it cleanly.
build_judge_prompt() {
    local prompt="$1"
    local expected="$2"
    local substrings="$3"
    local answer="$4"
    local chunks_text="$5"

    cat <<JUDGE_PROMPT
You are an impartial evaluator scoring a retrieval-augmented answer.

## Question asked
${prompt}

## Expected answer summary
${expected}

## Key substrings that should appear in the context (any one counts as a hit)
${substrings}

## Retrieved chunks (top-8, in ranked order)
${chunks_text}

## System answer
${answer}

---
Score the above on three axes, each from 1 (very poor) to 5 (excellent):

1. **chunk_relevance** — Do the retrieved chunks contain information that supports the question? (1 = chunks are entirely off-topic; 5 = chunks directly answer the question)
2. **answer_correctness** — How well does the system answer match the expected answer summary? (1 = contradicts or misses the point; 5 = fully correct and complete)
3. **coherence** — Is the answer well-structured and internally consistent? (1 = incoherent; 5 = clear, logically structured)

Respond with ONLY valid JSON (no markdown fences, no extra text):
{"relevance": <int>, "correctness": <int>, "coherence": <int>, "reasoning": "<one or two sentence justification>"}
JUDGE_PROMPT
}

# ── Judge loop ────────────────────────────────────────────────────────────────
> "${JUDGED_FILE}"
echo "Judging records from: ${INPUT_FILE}"
echo "Router: ${ROUTER_URL}   Model: ${MODEL}"
echo ""

TOTAL=0
JUDGED=0
SKIPPED=0

while IFS= read -r record; do
    (( TOTAL++ )) || true

    idx="$(echo "${record}"   | jq -r '.index')"
    mode_tok="$(echo "${record}" | jq -r '.mode_token')"
    prompt="$(echo "${record}" | jq -r '.prompt')"
    expected="$(echo "${record}" | jq -r '.expected_answer_summary // ""')"
    substrings="$(echo "${record}" | jq -r '.must_recall_substrings // ""')"
    answer="$(echo "${record}" | jq -r '.answer // ""')"
    http_status="$(echo "${record}" | jq -r '.http_status // 0')"

    # Skip failed/skipped records from the runner
    if [[ "${http_status}" != "200" ]]; then
        jq -c '. + {judge_skipped: true, judge_reason: "runner_error"}' \
            <<< "${record}" >> "${JUDGED_FILE}"
        (( SKIPPED++ )) || true
        echo "  [${idx}/${mode_tok}] SKIP (runner HTTP ${http_status})"
        continue
    fi

    # Build chunks text for the judge prompt
    chunks_text="$(echo "${record}" | jq -r \
        '[.chunks // [] | .[] | "### \(.title // "untitled")\n\(.excerpt // "")"] | join("\n\n")')"

    # If both answer and chunks are empty, skip (no useful signal)
    if [[ -z "${answer}" && -z "${chunks_text}" ]]; then
        jq -c '. + {judge_skipped: true, judge_reason: "empty_response"}' \
            <<< "${record}" >> "${JUDGED_FILE}"
        (( SKIPPED++ )) || true
        echo "  [${idx}/${mode_tok}] SKIP (empty answer + chunks)"
        continue
    fi

    judge_prompt="$(build_judge_prompt "${prompt}" "${expected}" "${substrings}" "${answer}" "${chunks_text}")"

    # Build OpenAI-compatible chat completions request.
    # chat_template_kwargs.enable_thinking=false disables thinking on Qwen3.6
    # hybrid models (vLLM ≥0.8.x with Qwen3 chat template). For models that
    # don't support this kwarg, vLLM ignores it gracefully.
    llm_body="$(jq -n \
        --arg model   "${MODEL}" \
        --arg content "${judge_prompt}" \
        '{
            model: $model,
            messages: [
                {role: "system", content: "You are an evaluation assistant. Always respond with valid JSON only."},
                {role: "user",   content: $content}
            ],
            temperature: 0.1,
            max_tokens: 1024,
            chat_template_kwargs: {enable_thinking: false}
        }')"

    printf "  [%02d/%s] judging … " "${idx}" "${mode_tok}"

    judge_response=""
    judge_http_status=0
    judge_raw="$(curl -s -w '\n__HTTP_STATUS__:%{http_code}' \
        --max-time 120 \
        -X POST \
        -H 'Content-Type: application/json' \
        -d "${llm_body}" \
        "${ROUTER_URL}/v1/chat/completions" 2>&1)" || true

    judge_http_status="$(echo "${judge_raw}" | grep '__HTTP_STATUS__:' | sed 's/__HTTP_STATUS__://')"
    judge_body="$(echo "${judge_raw}" | grep -v '__HTTP_STATUS__:')"

    if [[ "${judge_http_status}" == "200" ]]; then
        # Extract the content field from the first choice.
        # Qwen3.6 hybrid in thinking mode returns content=null + output in
        # .reasoning; /nothink should suppress this, but fall back just in case.
        judge_content="$(echo "${judge_body}" | jq -r '
            .choices[0].message.content
            // .choices[0].message.reasoning
            // ""
        ')"

        # Parse JSON scores from judge_content (tolerate minor whitespace)
        if echo "${judge_content}" | jq -e '.' > /dev/null 2>&1; then
            relevance="$(echo "${judge_content}"   | jq -r '.relevance   // 0')"
            correctness="$(echo "${judge_content}" | jq -r '.correctness // 0')"
            coherence="$(echo "${judge_content}"   | jq -r '.coherence   // 0')"
            reasoning="$(echo "${judge_content}"   | jq -r '.reasoning   // ""')"
            aggregate="$(echo "${relevance} ${correctness} ${coherence}" | \
                awk '{printf "%.2f", ($1+$2+$3)/3}')"

            jq -c \
                --argjson rel "${relevance}" \
                --argjson cor "${correctness}" \
                --argjson coh "${coherence}" \
                --arg rea "${reasoning}" \
                --arg agg "${aggregate}" \
                '. + {
                    judge_relevance:    $rel,
                    judge_correctness:  $cor,
                    judge_coherence:    $coh,
                    judge_aggregate:    ($agg | tonumber),
                    judge_reasoning:    $rea,
                    judge_skipped:      false
                }' <<< "${record}" >> "${JUDGED_FILE}"

            echo "OK (R=${relevance} C=${correctness} Coh=${coherence} avg=${aggregate})"
            (( JUDGED++ )) || true
        else
            # Judge returned non-JSON — store raw and mark parse error
            jq -c \
                --arg raw "${judge_content}" \
                '. + {judge_skipped: false, judge_error: "parse_failed", judge_raw: $raw,
                      judge_relevance: 0, judge_correctness: 0, judge_coherence: 0, judge_aggregate: 0}' \
                <<< "${record}" >> "${JUDGED_FILE}"
            echo "WARN (judge returned non-JSON): ${judge_content:0:60}…"
            (( JUDGED++ )) || true
        fi
    else
        jq -c \
            --argjson status "${judge_http_status:-0}" \
            '. + {judge_skipped: true, judge_reason: "judge_http_error", judge_http_status: $status}' \
            <<< "${record}" >> "${JUDGED_FILE}"
        echo "FAIL (judge HTTP ${judge_http_status})"
        (( SKIPPED++ )) || true
    fi

done < "${INPUT_FILE}"

echo ""
echo "Judging complete: ${JUDGED} judged, ${SKIPPED} skipped, ${TOTAL} total"
echo "Judged output: ${JUDGED_FILE}"

# ── Generate results report ───────────────────────────────────────────────────
echo ""
echo "Generating results report: ${RESULTS_FILE}"

python3 - <<'PYEOF' "${JUDGED_FILE}" "${RESULTS_FILE}"
import json, sys, collections, math
from datetime import date

judged_file  = sys.argv[1]
results_file = sys.argv[2]
today        = date.today().isoformat()

records = []
with open(judged_file) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Group by mode_token then by index
by_mode = collections.defaultdict(dict)
prompts_meta = {}  # index → {prompt, expected, substrings}

for r in records:
    idx  = r.get("index", 0)
    mode = r.get("mode_token", "unknown")
    by_mode[mode][idx] = r
    if idx not in prompts_meta:
        prompts_meta[idx] = {
            "prompt":   r.get("prompt", ""),
            "expected": r.get("expected_answer_summary", ""),
            "substrings": r.get("must_recall_substrings", ""),
        }

modes       = sorted(by_mode.keys())
all_indices = sorted(prompts_meta.keys())

# Per-mode aggregate stats
def mode_stats(mode_records):
    vals = [r for r in mode_records.values()
            if not r.get("judge_skipped") and r.get("judge_aggregate", 0) > 0]
    if not vals:
        return {"count": 0, "relevance": 0, "correctness": 0, "coherence": 0, "aggregate": 0}
    n = len(vals)
    return {
        "count":       n,
        "relevance":   sum(r.get("judge_relevance",   0) for r in vals) / n,
        "correctness": sum(r.get("judge_correctness", 0) for r in vals) / n,
        "coherence":   sum(r.get("judge_coherence",   0) for r in vals) / n,
        "aggregate":   sum(r.get("judge_aggregate",   0) for r in vals) / n,
    }

mode_aggs = {m: mode_stats(by_mode[m]) for m in modes}

# Compute Δ (deep − default) if both present
# Canonical mode pair: "hybrid" (=default) vs "hipporag" (=deep)
BASELINE_TOKEN = None
DEEP_TOKEN     = None
for m in modes:
    if m in ("hybrid", "default"):
        BASELINE_TOKEN = m
    if m in ("hipporag", "deep"):
        DEEP_TOKEN = m

# ── Write report ──────────────────────────────────────────────────────────────
lines = []
lines.append(f"# graphrag-rs eval results — {today}")
lines.append("")
lines.append("> Generated by `eval/judge.sh`. Re-run to refresh.")
lines.append("> Scoring model: LLM judge via `127.0.0.1:17170` (llm-router → Spark 1).")
lines.append("> Scores: 1 (worst) – 5 (best). Δ = deep − default (positive = improvement).")
lines.append("")

# ── Aggregate table ───────────────────────────────────────────────────────────
lines.append("## Aggregate scores by mode")
lines.append("")
lines.append("| Mode | N | Relevance | Correctness | Coherence | Aggregate |")
lines.append("|---|---|---|---|---|---|")
for m in modes:
    s = mode_aggs[m]
    lines.append(f"| {m} | {s['count']} | {s['relevance']:.2f} | {s['correctness']:.2f} | {s['coherence']:.2f} | {s['aggregate']:.2f} |")
lines.append("")

# Δ row
if BASELINE_TOKEN and DEEP_TOKEN and mode_aggs[BASELINE_TOKEN]["count"] > 0 and mode_aggs[DEEP_TOKEN]["count"] > 0:
    b = mode_aggs[BASELINE_TOKEN]
    d = mode_aggs[DEEP_TOKEN]
    lines.append("## Δ (deep − default)")
    lines.append("")
    lines.append("| Axis | Baseline | Deep | Δ | Δ % |")
    lines.append("|---|---|---|---|---|")
    for axis in ("relevance", "correctness", "coherence", "aggregate"):
        bv = b[axis]; dv = d[axis]
        delta = dv - bv
        delta_pct = (delta / bv * 100) if bv else 0
        marker = " ✅" if delta > 0 else (" ⚠️" if delta < -0.1 else "")
        lines.append(f"| {axis} | {bv:.2f} | {dv:.2f} | {delta:+.2f} | {delta_pct:+.1f}%{marker} |")
    lines.append("")

# ── Per-prompt table ──────────────────────────────────────────────────────────
lines.append("## Per-prompt scores")
lines.append("")

# Build column headers
mode_cols = modes
header_parts = ["#", "Prompt (truncated)"] + [f"{m} R/C/Coh/Avg" for m in mode_cols]
lines.append("| " + " | ".join(header_parts) + " |")
sep_parts = ["---"] * len(header_parts)
lines.append("| " + " | ".join(sep_parts) + " |")

for idx in all_indices:
    meta = prompts_meta[idx]
    short_prompt = meta["prompt"][:60].replace("|", "\\|") + ("…" if len(meta["prompt"]) > 60 else "")
    row = [str(idx), short_prompt]
    for m in mode_cols:
        r = by_mode[m].get(idx)
        if r and not r.get("judge_skipped") and r.get("judge_aggregate", 0) > 0:
            rel = r.get("judge_relevance",   "-")
            cor = r.get("judge_correctness", "-")
            coh = r.get("judge_coherence",   "-")
            agg = r.get("judge_aggregate",   "-")
            cell = f"{rel}/{cor}/{coh}/{agg:.2f}" if isinstance(agg, float) else f"{rel}/{cor}/{coh}/{agg}"
        elif r and r.get("judge_skipped"):
            cell = f"skip ({r.get('judge_reason', '?')})"
        else:
            cell = "—"
        row.append(cell)
    lines.append("| " + " | ".join(row) + " |")

lines.append("")

# ── Per-prompt reasoning ──────────────────────────────────────────────────────
lines.append("## Per-prompt judge reasoning")
lines.append("")
for idx in all_indices:
    meta = prompts_meta[idx]
    lines.append(f"### Prompt {idx}")
    lines.append(f"**Q:** {meta['prompt']}")
    lines.append(f"**Expected:** {meta['expected'][:200]}…")
    lines.append(f"**Substrings:** `{meta['substrings']}`")
    lines.append("")
    for m in modes:
        r = by_mode[m].get(idx)
        if r and not r.get("judge_skipped") and r.get("judge_reasoning"):
            reasoning = r["judge_reasoning"]
            lines.append(f"- **{m}**: {reasoning}")
    lines.append("")

with open(results_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Report written: {results_file}")
PYEOF

echo ""
echo "Done. Results: ${RESULTS_FILE}"
