#!/usr/bin/env bash
# eval/run-eval.sh — graphrag-rs LLM-judge eval harness runner
#
# POSTs each prompt from hipporag-vs-baseline.csv against graphrag-server
# in each configured MODE and captures chunks + final answer per variant.
# Output: JSONL file at eval/raw-<DATE>.jsonl (one record per prompt×mode).
#
# Usage:
#   ./eval/run-eval.sh [--modes MODE1,MODE2] [--server URL] [--csv FILE] [--out FILE]
#
# Defaults:
#   --modes   hybrid,hipporag    (pre-card-3: only 'hybrid' will succeed)
#   --server  http://127.0.0.1:17180
#   --csv     eval/hipporag-vs-baseline.csv  (relative to repo root)
#   --out     eval/raw-<YYYY-MM-DD>.jsonl
#
# Mode mapping (mirrors memory-mcp convention):
#   "default" token → server mode "hybrid"
#   "deep"    token → server mode "hipporag"  (available after card 3)
#   Any other token → passed through verbatim
#
# The harness is intentionally GENERIC over modes. Future phases (BM25
# fusion, query decomposition) add new mode tokens and rerun without
# editing this script.
#
# Requirements: bash ≥4, curl, jq, python3 (for CSV parsing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve python3: prefer PATH, fall back to nix store glob (direnv strips
# PATH in the graphrag-rs dev shell, which has no Python derivation).
# Use glob expansion (/nix/store/*python3-*-env/bin/python3) — far faster
# than `find /nix/store` on a large store.
if ! command -v python3 &>/dev/null; then
    for _candidate in /nix/store/*python3-3*-env/bin/python3; do
        if [[ -x "${_candidate}" ]]; then
            export PATH="$(dirname "${_candidate}"):${PATH}"
            break
        fi
    done
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
MODES="hybrid,hipporag"
SERVER_URL="http://127.0.0.1:17180"
CSV_FILE="${SCRIPT_DIR}/hipporag-vs-baseline.csv"
DATE_STAMP="$(date +%Y-%m-%d)"
OUT_FILE="${SCRIPT_DIR}/raw-${DATE_STAMP}.jsonl"
TOP_K=10

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes)   MODES="$2";      shift 2 ;;
        --server)  SERVER_URL="$2"; shift 2 ;;
        --csv)     CSV_FILE="$2";   shift 2 ;;
        --out)     OUT_FILE="$2";   shift 2 ;;
        --top-k)   TOP_K="$2";      shift 2 ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | head -30
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Mode token → server mode mapping ─────────────────────────────────────────
# Normalise the client-facing mode tokens to graphrag-server mode values.
map_mode() {
    local token="$1"
    case "${token}" in
        default)  echo "hybrid"   ;;
        deep)     echo "hipporag" ;;
        *)        echo "${token}" ;;   # pass through (local, global, mix, …)
    esac
}

# ── Pre-flight: check dependencies ───────────────────────────────────────────
for cmd in curl jq python3; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: required command '${cmd}' not found on PATH." >&2
        exit 1
    fi
done

if [[ ! -f "${CSV_FILE}" ]]; then
    echo "Error: CSV file not found: ${CSV_FILE}" >&2
    exit 1
fi

# ── Pre-flight: check graphrag-server health ──────────────────────────────────
echo "Checking graphrag-server at ${SERVER_URL} …"
if ! curl -fsS --max-time 5 "${SERVER_URL}/api/graph/stats" > /dev/null 2>&1; then
    echo "Warning: graphrag-server health check failed — it may be starting up." >&2
    echo "         Proceeding anyway; individual POSTs will fail if the server is down." >&2
fi

# ── Parse modes list ──────────────────────────────────────────────────────────
IFS=',' read -ra MODE_TOKENS <<< "${MODES}"

# ── Output setup ─────────────────────────────────────────────────────────────
# Truncate output file for this run (idempotent re-run).
> "${OUT_FILE}"
echo "Output: ${OUT_FILE}"
echo "Modes:  ${MODE_TOKENS[*]}"
echo ""

# ── CSV → prompt records (skip header) ───────────────────────────────────────
# Use python3 to parse CSV properly (handles quoted commas in prompts).
PROMPT_JSON="$(python3 - "${CSV_FILE}" <<'PYEOF'
import csv, json, sys

rows = []
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        rows.append({
            "index": i + 1,
            "prompt": row["prompt"],
            "expected_answer_summary": row["expected_answer_summary"],
            "must_recall_substrings": row["must_recall_substrings"],
        })

print(json.dumps(rows))
PYEOF
)"

TOTAL_PROMPTS="$(echo "${PROMPT_JSON}" | jq 'length')"
echo "Loaded ${TOTAL_PROMPTS} prompts from ${CSV_FILE}"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0

for mode_token in "${MODE_TOKENS[@]}"; do
    server_mode="$(map_mode "${mode_token}")"
    echo "=== Mode: ${mode_token} (server: ${server_mode}) ==="

    # Iterate over all prompts
    while IFS= read -r record; do
        idx="$(echo "${record}"   | jq -r '.index')"
        prompt="$(echo "${record}" | jq -r '.prompt')"
        expected="$(echo "${record}" | jq -r '.expected_answer_summary')"
        substrings="$(echo "${record}" | jq -r '.must_recall_substrings')"

        printf "  [%02d/%02d] %s … " "${idx}" "${TOTAL_PROMPTS}" \
            "$(echo "${prompt}" | head -c 60)…"

        # Build the POST body
        body="$(jq -n \
            --arg q "${prompt}" \
            --arg m "${server_mode}" \
            --argjson k "${TOP_K}" \
            '{query: $q, mode: $m, topK: $k}')"

        # POST to graphrag-server (60s timeout — LLM synthesis can be slow)
        http_status=0
        response=""
        response="$(curl -s -w '\n__HTTP_STATUS__:%{http_code}' \
            --max-time 90 \
            -X POST \
            -H 'Content-Type: application/json' \
            -d "${body}" \
            "${SERVER_URL}/api/query" 2>&1)" || true

        # Split response body and HTTP status
        http_status="$(echo "${response}" | grep '__HTTP_STATUS__:' | sed 's/__HTTP_STATUS__://')"
        body_text="$(echo "${response}" | grep -v '__HTTP_STATUS__:')"

        if [[ "${http_status}" == "200" ]]; then
            # Extract fields we care about
            answer="$(echo "${body_text}" | jq -r '.answer // ""')"
            results_count="$(echo "${body_text}" | jq '[.results // [] | .[] | .excerpt] | length')"
            # Collect first N excerpts as the "chunks" for judging
            chunks_json="$(echo "${body_text}" | jq '[.results // [] | .[] | {title, excerpt, similarity}] | .[0:8]')"
            elapsed_ms="$(echo "${body_text}" | jq -r '.processingTimeMs // 0')"

            # Emit JSONL record (compact — one line per record required for JSONL)
            jq -cn \
                --argjson  idx       "${idx}" \
                --arg      prompt    "${prompt}" \
                --arg      mode_tok  "${mode_token}" \
                --arg      srv_mode  "${server_mode}" \
                --arg      expected  "${expected}" \
                --arg      substrings "${substrings}" \
                --arg      answer    "${answer}" \
                --argjson  chunks    "${chunks_json}" \
                --argjson  elapsed   "${elapsed_ms}" \
                '{
                    index:          $idx,
                    prompt:         $prompt,
                    mode_token:     $mode_tok,
                    server_mode:    $srv_mode,
                    expected_answer_summary: $expected,
                    must_recall_substrings:  $substrings,
                    answer:         $answer,
                    chunks:         $chunks,
                    elapsed_ms:     $elapsed,
                    http_status:    200
                }' >> "${OUT_FILE}"

            echo "OK (${elapsed_ms}ms, ${results_count} chunks)"
            (( PASS++ )) || true
        else
            # Emit error record (compact — one line per record)
            jq -cn \
                --argjson  idx       "${idx}" \
                --arg      prompt    "${prompt}" \
                --arg      mode_tok  "${mode_token}" \
                --arg      srv_mode  "${server_mode}" \
                --argjson  status    "${http_status:-0}" \
                --arg      body      "${body_text}" \
                '{
                    index:       $idx,
                    prompt:      $prompt,
                    mode_token:  $mode_tok,
                    server_mode: $srv_mode,
                    http_status: $status,
                    error:       $body,
                    answer:      null,
                    chunks:      []
                }' >> "${OUT_FILE}"

            if [[ "${server_mode}" == "hipporag" && ( "${http_status}" == "422" || "${http_status}" == "400" ) ]]; then
                echo "SKIP (mode not yet live — expected before card 3)"
                (( SKIP++ )) || true
            else
                echo "FAIL (HTTP ${http_status})"
                (( FAIL++ )) || true
            fi
        fi

    done < <(echo "${PROMPT_JSON}" | jq -c '.[]')

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_REQUESTS=$(( PASS + FAIL + SKIP ))
echo "────────────────────────────────────────"
echo "Run complete: ${TOTAL_REQUESTS} requests"
echo "  OK:    ${PASS}"
echo "  FAIL:  ${FAIL}"
echo "  SKIP:  ${SKIP}  (hipporag mode not yet deployed — expected)"
echo ""
echo "Raw output: ${OUT_FILE}"
echo "Next step:  ./eval/judge.sh --input ${OUT_FILE}"
