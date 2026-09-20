#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G"
export SPRING_VACE_ROOT="${SPRING_VACE_ROOT:-/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_480x480}"
INPUT_EXPORT="${SPRING_REUSE_INPUT_ROOT:-/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_480x832}"
exec bash "$PROJECT_ROOT/scripts/run_spring_vace_export.sh" --height 480 --width 480 --context-count 17 --fixed-bounds --cached-rgb-from "$INPUT_EXPORT" "$@"
