#!/usr/bin/env bash
# Run ct87.prepare_data with an RSS sampler alongside the python process.
# Usage:
#   run_prep_with_rss.sh <output-dir> <max-tokens> [seq-len] [val-fraction] [sample-interval-sec]
# Outputs:
#   training/logs/prep_<tag>.log          — python stdout/stderr
#   training/logs/prep_<tag>_rss.csv      — ts,rss_kb,vmhwm_kb,vmpeak_kb,vmsize_kb,state
set -eu

OUT_DIR="${1:?output dir required}"
MAX_TOKENS="${2:?max-tokens required}"
SEQ_LEN="${3:-2048}"
VAL_FRAC="${4:-0.01}"
INTERVAL="${5:-3}"

WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT" || exit 1

TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="training/logs"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/prep_${TAG}.log"
RSS_CSV="$LOG_DIR/prep_${TAG}_rss.csv"

echo "ts,rss_kb,vmhwm_kb,vmpeak_kb,vmsize_kb,state" > "$RSS_CSV"

training/.venv/bin/python -u -m ct87.prepare_data \
    --output "$OUT_DIR" \
    --seq-len "$SEQ_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --val-fraction "$VAL_FRAC" \
    > "$STDOUT_LOG" 2>&1 &
PID=$!

# Forward INT/TERM to the background prep. Without this, Ctrl-C on a
# multi-hour run exits the wrapper shell but leaves ct87.prepare_data
# orphaned and still consuming CPU/network. Disable the trap inside the
# handler so a second signal can force the wrapper to quit immediately.
_forward_signal() {
    trap - INT TERM
    if [[ -n "${PID:-}" ]]; then
        kill -TERM "$PID" 2>/dev/null || true
    fi
}
trap _forward_signal INT TERM

echo "PID=$PID"
echo "STDOUT=$STDOUT_LOG"
echo "RSS=$RSS_CSV"

while kill -0 "$PID" 2>/dev/null; do
    if [[ -r "/proc/$PID/status" ]]; then
        ts="$(date +%s)"
        st="$(cat "/proc/$PID/status" 2>/dev/null || true)"
        rss="$(awk '/^VmRSS:/  {print $2}' <<<"$st")"
        hwm="$(awk '/^VmHWM:/  {print $2}' <<<"$st")"
        peak="$(awk '/^VmPeak:/ {print $2}' <<<"$st")"
        size="$(awk '/^VmSize:/ {print $2}' <<<"$st")"
        state="$(awk '/^State:/  {print $2}' <<<"$st")"
        if [[ -n "$rss" ]]; then
            echo "$ts,$rss,$hwm,$peak,$size,$state" >> "$RSS_CSV"
        fi
    fi
    sleep "$INTERVAL"
done

# Wrap `wait` in an if/else so `set -e` doesn't abort the script when the
# prep exits non-zero — we still want to log the exit code and close out
# the CSV for post-mortem analysis of a failed run.
if wait "$PID"; then
    RC=0
else
    RC=$?
fi

# Emit the exit-code line as a valid CSV row matching the header so
# downstream parsers (pandas, csv.DictReader) don't choke. All metric
# columns are empty; the exit code goes into the `state` column as
# `EXIT:<rc>` which is distinguishable from the single-char process
# states (R/S/D/Z) above.
echo "$(date +%s),,,,,EXIT:$RC" >> "$RSS_CSV"
echo "DONE rc=$RC"
exit "$RC"
