#!/bin/bash

# Usage:
#   bash tools/auto_resume_train.sh teacher <exptid> <device>
#   bash tools/auto_resume_train.sh student <student_id> <teacher_id> <device>
#
# Examples:
#   bash tools/auto_resume_train.sh teacher 0523_twist_teacher cuda:0
#   bash tools/auto_resume_train.sh student 0523_twist_rlbcstu 0523_twist_teacher cuda:0

MODE=$1

if [ "$MODE" != "teacher" ] && [ "$MODE" != "student" ] && [ "$MODE" != "cleaned" ]; then
    echo "Usage: bash tools/auto_resume_train.sh <teacher|student|cleaned> <args...>"
    echo ""
    echo "  teacher: bash tools/auto_resume_train.sh teacher <exptid> <device>"
    echo "  student: bash tools/auto_resume_train.sh student <student_id> <teacher_id> <device>"
    echo "  cleaned: bash tools/auto_resume_train.sh cleaned <student_id> <teacher_id> <device>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/tools/auto_resume_${MODE}.log"
TRAIN_DIR="$PROJECT_DIR/legged_gym/legged_gym/scripts"

RESTART_COUNT=0
MAX_RESTARTS=100
RESTART_DELAY=30

# Check if model files exist for this experiment
has_checkpoint() {
    local proj=$1
    local exptid=$2
    local log_dir="$PROJECT_DIR/legged_gym/logs/${proj}/${exptid}"
    if [ -d "$log_dir" ] && ls "$log_dir"/model_*.pt 2>/dev/null | head -1 | grep -q .; then
        return 0
    fi
    return 1
}

if [ "$MODE" = "teacher" ]; then
    EXPTID=$2
    DEVICE=$3
    TASK="g1_priv_mimic"
    PROJ="g1_priv_mimic"
    TEACHER_ARGS=""
elif [ "$MODE" = "cleaned" ]; then
    STUDENT_ID=$2
    TEACHER_ID=$3
    DEVICE=$4
    TASK="g1_stu_rl_cleaned"
    PROJ="g1_stu_rl"
    TEACHER_ARGS="--teacher_exptid ${TEACHER_ID} --teacher_checkpoint 42500"
    EXPTID=$STUDENT_ID
else
    STUDENT_ID=$2
    TEACHER_ID=$3
    DEVICE=$4
    TASK="g1_stu_rl"
    PROJ="g1_stu_rl"
    TEACHER_ARGS="--teacher_exptid ${TEACHER_ID}"
    EXPTID=$STUDENT_ID
fi

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training (restart #$RESTART_COUNT)" | tee -a "$LOG_FILE"

    # Kill any leftover train.py processes for this experiment to free GPU memory
    STALE_PIDS=$(pgrep -f "train.py.*--exptid ${EXPTID}" 2>/dev/null || true)
    if [ -n "$STALE_PIDS" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing leftover train.py processes for ${EXPTID}: $STALE_PIDS" | tee -a "$LOG_FILE"
        kill $STALE_PIDS 2>/dev/null || true
        sleep 5
        kill -9 $STALE_PIDS 2>/dev/null || true
        sleep 5
    fi

    cd "$TRAIN_DIR"

    if [ $RESTART_COUNT -eq 0 ] && has_checkpoint "$PROJ" "$EXPTID"; then
        RESUME_FLAGS="--resume --resumeid ${EXPTID}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found existing checkpoints, resuming from latest" | tee -a "$LOG_FILE"
    elif [ $RESTART_COUNT -gt 0 ]; then
        RESUME_FLAGS="--resume --resumeid ${EXPTID}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-resuming after crash" | tee -a "$LOG_FILE"
    else
        RESUME_FLAGS=""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fresh start" | tee -a "$LOG_FILE"
    fi

    set +e
    python train.py \
        --task "$TASK" \
        --proj_name "$PROJ" \
        --exptid "$EXPTID" \
        --device "$DEVICE" \
        $TEACHER_ARGS \
        $RESUME_FLAGS
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished normally." | tee -a "$LOG_FILE"
        exit 0
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training crashed (exit code=$EXIT_CODE), restarting in ${RESTART_DELAY}s (attempt $RESTART_COUNT/$MAX_RESTARTS)" | tee -a "$LOG_FILE"
    sleep $RESTART_DELAY
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reached max restarts ($MAX_RESTARTS), giving up." | tee -a "$LOG_FILE"
exit 1
