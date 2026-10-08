#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
PYTHON="/home/yangx/miniconda3/envs/geo_v2/bin/python"
POINT_COUNT="${P2P_FPS_POINTS:-1024}"
FORCE="${P2P_FORCE:-0}"
DRY_RUN="${P2P_DRY_RUN:-0}"
BASE_ROOT="$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_correspondence_best3"
BASE_SELECTION="$BASE_ROOT/selection.json"
OUTPUT_ROOT="${P2P_FPS_OUTPUT_ROOT:-$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_correspondence_best3_fps${POINT_COUNT}}"
SELECTION="$OUTPUT_ROOT/selection.json"

printf 'PREPARE_FPS count=%s force=%s output=%s base_selection=%s prepared_selection=1\n' \
  "$POINT_COUNT" "$FORCE" "$OUTPUT_ROOT" "$BASE_SELECTION"

if [[ "$DRY_RUN" == "1" ]]; then
  P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
  P2P_PREPARED_SELECTION=1 \
  P2P_DRY_RUN=1 \
  bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"
  printf 'RENDER_FPS dataset=all output=%s\n' "$OUTPUT_ROOT"
  exit 0
fi

if [[ ! -s "$SELECTION" || "$FORCE" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/prepare_visibility_fps.py" \
    --selection "$BASE_SELECTION" \
    --output-root "$OUTPUT_ROOT" \
    --count "$POINT_COUNT"
fi

P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
P2P_PREPARED_SELECTION=1 \
bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"

"$PYTHON" "$PROJECT_ROOT/visualization/correspondence_method_comparison.py" \
  --selection "$SELECTION" \
  --output-root "$OUTPUT_ROOT" \
  --dataset all \
  --dpi 300

printf 'RENDER_FPS dataset=all output=%s\n' "$OUTPUT_ROOT"
