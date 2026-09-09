#!/usr/bin/env bash

# Auto-resume TWIST-AnyAdapter safe training.
#
# Usage:
#   bash tools/auto_resume_anyadapter_safe.sh [exptid] [device] [num_envs] [max_iterations] [extra train.py args...]
#
# Examples:
#   bash tools/auto_resume_anyadapter_safe.sh
#   bash tools/auto_resume_anyadapter_safe.sh twist_anyadapter_safe_train cuda:0
#   bash tools/auto_resume_anyadapter_safe.sh twist_anyadapter_safe_1000it cuda:0 1024 1000 --no_wandb
#
# This script only launches simulation training. It does not run deploy_real or Unitree code.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/legged_gym/legged_gym/scripts/train.py"

TASK="g1_stu_anyadapter_safe"
PROJ="${PROJ:-g1_twist_anyadapter_safe}"
EXPTID="${1:-twist_anyadapter_safe_train}"
DEVICE="${2:-cuda:0}"
NUM_ENVS="${3:-}"
MAX_ITERATIONS="${4:-}"

shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ARGS=("$@")

LOG_FILE="$PROJECT_DIR/tools/auto_resume_anyadapter_safe_${EXPTID}.log"
RUN_DIR="$PROJECT_DIR/legged_gym/logs/${PROJ}/${EXPTID}"

MAX_RESTARTS="${MAX_RESTARTS:-100}"
RESTART_DELAY="${RESTART_DELAY:-30}"
RESTART_COUNT=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

has_checkpoint() {
    [ -d "$RUN_DIR" ] && find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -type f | grep -q .
}

latest_checkpoint() {
    if ! has_checkpoint; then
        echo "none"
        return
    fi
    find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -type f \
        | sed -E 's/.*model_([0-9]+)\.pt/\1/' \
        | sort -n \
        | tail -1
}

build_train_cmd() {
    TRAIN_CMD=(
        python "$TRAIN_SCRIPT"
        --task "$TASK"
        --proj_name "$PROJ"
        --exptid "$EXPTID"
        --run_name "$EXPTID"
        --device "$DEVICE"
    )

    if [ -n "$NUM_ENVS" ]; then
        TRAIN_CMD+=(--num_envs "$NUM_ENVS")
    fi
    if [ -n "$MAX_ITERATIONS" ]; then
        TRAIN_CMD+=(--max_iterations "$MAX_ITERATIONS")
    fi

    if has_checkpoint; then
        TRAIN_CMD+=(--resume --resumeid "$EXPTID")
    fi

    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
}

log "TWIST-AnyAdapter safe auto-resume started"
log "task=${TASK} proj=${PROJ} exptid=${EXPTID} device=${DEVICE}"
log "run_dir=${RUN_DIR}"
log "max_restarts=${MAX_RESTARTS} restart_delay=${RESTART_DELAY}s"

while [ "$RESTART_COUNT" -lt "$MAX_RESTARTS" ]; do
    if has_checkpoint; then
        log "Found checkpoint model_$(latest_checkpoint).pt; resume enabled"
    else
        log "No checkpoint found; fresh start"
    fi

    STALE_PIDS="$(pgrep -f "train.py.*--exptid ${EXPTID}" 2>/dev/null || true)"
    if [ -n "$STALE_PIDS" ]; then
        log "Killing leftover train.py processes for ${EXPTID}: ${STALE_PIDS}"
        kill $STALE_PIDS 2>/dev/null || true
        sleep 5
        kill -9 $STALE_PIDS 2>/dev/null || true
        sleep 5
    fi

    build_train_cmd
    log "Launching: ${TRAIN_CMD[*]}"

    "${TRAIN_CMD[@]}"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        log "Training finished normally."
        exit 0
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    log "Training exited with code=${EXIT_CODE}; restarting in ${RESTART_DELAY}s (${RESTART_COUNT}/${MAX_RESTARTS})"
    sleep "$RESTART_DELAY"
done

log "Reached max restarts (${MAX_RESTARTS}); giving up."
exit 1
