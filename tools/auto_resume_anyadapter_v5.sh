#!/usr/bin/env bash

# Fixed TWIST-AnyAdapter V5 heading-aware training launcher.
# Runs simulation training only; it never invokes deploy_real or Unitree code.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/home/hank/anaconda3/envs/twist/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/legged_gym/legged_gym/scripts/train.py"

TASK="g1_stu_anyadapter_v5"
PROJ="g1_twist_anyadapter_v5_heading_0529"
EXPTID="twist_anyadapter_v5_heading_0529_train"
DEVICE="cuda:0"
NUM_ENVS=4096
TARGET_ITERATIONS=30000
MAX_RESTARTS=100
RESTART_DELAY=30

RUN_DIR="$PROJECT_DIR/legged_gym/logs/$PROJ/$EXPTID"
LOG_FILE="$PROJECT_DIR/tools/auto_resume_anyadapter_v5.log"
LOCK_FILE="$PROJECT_DIR/tools/auto_resume_anyadapter_v5.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another V5 auto-resume launcher is already running." >&2
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

latest_iteration() {
    if [ ! -d "$RUN_DIR" ]; then
        echo 0
        return
    fi
    local latest
    latest="$(
        find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -type f -printf '%f\n' \
            | sed -E 's/model_([0-9]+)\.pt/\1/' \
            | sort -n \
            | tail -n 1
    )"
    echo "${latest:-0}"
}

log "TWIST-AnyAdapter V5 heading-aware auto-resume started"
log "task=$TASK proj=$PROJ exptid=$EXPTID"
log "device=$DEVICE num_envs=$NUM_ENVS target_iterations=$TARGET_ITERATIONS"
log "run_dir=$RUN_DIR"

restart_count=0
while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    current="$(latest_iteration)"
    if [ "$current" -ge "$TARGET_ITERATIONS" ]; then
        log "Target reached at model_${current}.pt."
        exit 0
    fi

    remaining=$((TARGET_ITERATIONS - current))
    train_cmd=(
        "$PYTHON_BIN" "$TRAIN_SCRIPT"
        --task "$TASK"
        --proj_name "$PROJ"
        --exptid "$EXPTID"
        --run_name "$EXPTID"
        --device "$DEVICE"
        --num_envs "$NUM_ENVS"
        --max_iterations "$remaining"
        --no_wandb
    )
    if [ "$current" -gt 0 ]; then
        train_cmd+=(--resume --resumeid "$EXPTID")
        log "Resuming model_${current}.pt; remaining_iterations=$remaining"
    else
        log "Starting a fresh 0529-based V5 run; iterations=$remaining"
    fi

    "${train_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}

    current="$(latest_iteration)"
    if [ "$current" -ge "$TARGET_ITERATIONS" ]; then
        log "Training completed at model_${current}.pt."
        exit 0
    fi

    restart_count=$((restart_count + 1))
    log "Training exited with code=$exit_code at iteration=$current; restarting in ${RESTART_DELAY}s ($restart_count/$MAX_RESTARTS)"
    sleep "$RESTART_DELAY"
done

log "Reached max restarts ($MAX_RESTARTS) before target iteration."
exit 1
