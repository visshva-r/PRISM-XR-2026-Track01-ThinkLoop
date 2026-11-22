#!/usr/bin/env bash
# Usage: ./run.sh [all|left|right|motion|clockwise|directions|textures|build] [extra eyes.py args]
# Requires BLENDER_EXE pointing at the blender binary (Blender 4.2+).
set -euo pipefail
cd "$(dirname "$0")"
BLENDER_EXE="${BLENDER_EXE:-blender}"
TASK="${1:-all}"
shift || true
case "$TASK" in
  all)
    python3 code/texture_synth.py
    "$BLENDER_EXE" -b --python code/build_scene.py
    "$BLENDER_EXE" -b scene.blend --python wrapper/eyes.py -- --input all ;;
  build)
    python3 code/texture_synth.py
    "$BLENDER_EXE" -b --python code/build_scene.py ;;
  *)
    "$BLENDER_EXE" -b scene.blend --python wrapper/eyes.py -- --input "$TASK" "$@" ;;
esac
