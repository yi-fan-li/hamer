#!/bin/bash
# watch_train.sh — auto-restart training on dataloader / worker crashes.
#
# Usage:
#   bash watch_train.sh
#
# The training command always exits 0 even on crash, so we detect failure by
# scanning the captured output for known crash patterns.
# Training resumes automatically from last.ckpt on each relaunch.

MAX_RETRIES=20
RETRY_DELAY=15      # seconds to wait between restarts
MIN_RUNTIME=60      # if training dies in under this many seconds, stop immediately
                    # (persistent bug, not a transient worker crash)

# CMD="python train.py exp_name=hamer data=mix_all experiment=hamer_vitpose_base trainer=gpu launcher=local"
CMD="python train_distill.py exp_name=distill_vitb experiment=hamer_distill trainer=gpu launcher=local     DISTILL.TEACHER_CHECKPOINT=_DATA/hamer_ckpts/checkpoints/hamer.ckpt     DISTILL.STUDENT_CHECKPOINT=_DATA/hamer_ckpts/hamer_base/checkpoints/hamer_base.ckpt"
RESTART_LOG="watch_train_restarts.log"

# Patterns in stdout/stderr that signal a crash
CRASH_PATTERNS=(
    "killed by signal"
    "ConnectionResetError"
    "RuntimeError: DataLoader worker"
    "During handling of the above exception, another exception occurred"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESTART_LOG"; }

log "Starting watch_train.sh  (max retries: $MAX_RETRIES)"

for attempt in $(seq 1 $MAX_RETRIES); do
    log "=== Attempt $attempt / $MAX_RETRIES ==="

    tmplog=$(mktemp /tmp/hamer_train_XXXXX.log)
    start_ts=$SECONDS

    # Stream output to the terminal in real-time while capturing for crash detection.
    # The temp log is deleted after each attempt so disk usage stays bounded.
    $CMD 2>&1 | tee "$tmplog"

    runtime=$(( SECONDS - start_ts ))

    # Check for any crash pattern in the captured output
    crashed=0
    crash_reason=""
    for pattern in "${CRASH_PATTERNS[@]}"; do
        if grep -qF "$pattern" "$tmplog"; then
            crashed=1
            crash_reason="$pattern"
            break
        fi
    done

    rm -f "$tmplog"

    if [ $crashed -eq 0 ]; then
        log "Training completed successfully (no crash pattern found)."
        exit 0
    fi

    log "Crash detected after ${runtime}s — matched: '$crash_reason'"

    # If it died very quickly the problem is probably not a transient worker crash,
    # so retrying would just loop forever on the same error.
    if [ $runtime -lt $MIN_RUNTIME ]; then
        log "Died in under ${MIN_RUNTIME}s — likely a persistent error. Stopping."
        exit 1
    fi

    if [ $attempt -lt $MAX_RETRIES ]; then
        log "Relaunching in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    fi
done

log "Reached max retries ($MAX_RETRIES). Giving up."
exit 1
