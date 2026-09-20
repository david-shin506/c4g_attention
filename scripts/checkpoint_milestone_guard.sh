#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_DIR ARCHIVE_DIR TRAIN_PGID" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
ARCHIVE_DIR="$2"
TRAIN_PGID="$3"

PRESERVE_STEP="${PRESERVE_STEP:-20000}"
STOP_STEP="${STOP_STEP:-30000}"
POLL_SECONDS="${POLL_SECONDS:-15}"
STABLE_POLLS="${STABLE_POLLS:-2}"
MIN_CHECKPOINT_BYTES="${MIN_CHECKPOINT_BYTES:-1000000000}"
GRACE_SECONDS="${GRACE_SECONDS:-60}"

if [[ ! "$TRAIN_PGID" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_PGID must be a positive integer, got: $TRAIN_PGID" >&2
  exit 2
fi

mkdir -p "$ARCHIVE_DIR"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" >&2
}

group_alive() {
  ps -eo pgid=,stat= | awk -v target="$TRAIN_PGID" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

wait_for_stable_checkpoint() {
  local step="$1"
  local checkpoint=""
  local size=-1
  local previous_size=-2
  local stable_count=0

  log "waiting for step ${step} checkpoint"
  while true; do
    if ! group_alive; then
      log "training process group ${TRAIN_PGID} stopped before step ${step}"
      return 1
    fi

    checkpoint="$(
      find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
        -name "*-step_${step}.ckpt" -print -quit
    )"
    if [[ -n "$checkpoint" ]]; then
      size="$(stat -c '%s' "$checkpoint")"
      if (( size >= MIN_CHECKPOINT_BYTES && size == previous_size )); then
        stable_count=$((stable_count + 1))
      else
        stable_count=0
      fi
      previous_size="$size"

      if (( stable_count >= STABLE_POLLS )); then
        log "step ${step} checkpoint is stable (${size} bytes): ${checkpoint}"
        printf '%s\n' "$checkpoint"
        return 0
      fi
    fi

    sleep "$POLL_SECONDS"
  done
}

preserve_checkpoint() {
  local source="$1"
  local destination="${ARCHIVE_DIR}/$(basename "$source")"

  if [[ -e "$destination" ]]; then
    log "milestone already preserved: ${destination}"
    return 0
  fi

  if ln -- "$source" "$destination"; then
    log "preserved milestone with hard link: ${destination}"
  else
    cp --reflink=auto -- "$source" "$destination"
    log "preserved milestone with copy: ${destination}"
  fi
}

preserve_checkpoint "$(wait_for_stable_checkpoint "$PRESERVE_STEP")"
preserve_checkpoint "$(wait_for_stable_checkpoint "$STOP_STEP")"

log "step ${STOP_STEP} checkpoint is safe; sending SIGTERM to process group ${TRAIN_PGID}"
kill -TERM -- "-${TRAIN_PGID}"

grace_polls=$((GRACE_SECONDS / POLL_SECONDS))
for ((poll = 0; poll < grace_polls; poll++)); do
  if ! group_alive; then
    log "training process group exited cleanly"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

log "process group did not exit within ${GRACE_SECONDS}s; sending SIGKILL"
kill -KILL -- "-${TRAIN_PGID}" || true
