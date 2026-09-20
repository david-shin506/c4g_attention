#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SUPERSPLAT_URL="https://superspl.at/editor"

open "$SCRIPT_DIR"
open "$SUPERSPLAT_URL"

osascript <<'APPLESCRIPT'
display dialog "SuperSplat을 열었습니다. Finder에서 원하는 gaussian_sequence 폴더를 브라우저의 SuperSplat 화면으로 drag & drop하세요. frame_0000.ply, frame_0001.ply, ...가 자동으로 하나의 Timeline sequence로 인식됩니다." buttons {"확인"} default button "확인" with title "C4G Gaussian Sequence"
APPLESCRIPT
