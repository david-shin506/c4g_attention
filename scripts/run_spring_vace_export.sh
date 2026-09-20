#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G"
PYTHON_BIN="${SPRING_EXPORT_PYTHON:-/home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python}"
EXPORT_ROOT="${SPRING_VACE_ROOT:-/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_480x832}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u scripts/export_spring_vace.py --output "$EXPORT_ROOT" "$@"
"$PYTHON_BIN" -u scripts/validate_spring_vace_export.py "$EXPORT_ROOT"
