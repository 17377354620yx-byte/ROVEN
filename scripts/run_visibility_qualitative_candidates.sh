#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
PYTHON="/home/yangx/miniconda3/envs/geo_v2/bin/python"
RESULT_ROOT="$PROJECT_ROOT/results/visibility_0.2_0.4_benchmark_metrics_v2"
SILICO_ROOT="/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4"
VITRO_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
OUTPUT_ROOT="${P2P_QUALITATIVE_ROOT:-$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_qualitative_candidates60_fps1024}"
PRE_FPS_ROOT="$OUTPUT_ROOT/pre_fps"
CANDIDATE_COUNT="${P2P_CANDIDATE_COUNT:-60}"
PER_METRIC="${P2P_CANDIDATES_PER_METRIC:-20}"
POINT_COUNT="${P2P_FPS_POINTS:-1024}"
TOP_K="${P2P_FINAL_TOP_K:-3}"
FORCE="${P2P_FORCE:-0}"
DRY_RUN="${P2P_DRY_RUN:-0}"

printf 'SELECT_CANDIDATES count=%s per_metric=%s output=%s\n' \
  "$CANDIDATE_COUNT" "$PER_METRIC" "$PRE_FPS_ROOT"
printf 'PREPARE_FPS count=%s output=%s\n' "$POINT_COUNT" "$OUTPUT_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'RUN_MODELS P2P_PREPARED_SELECTION=1 output=%s\n' "$OUTPUT_ROOT"
  P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
  P2P_PREPARED_SELECTION=1 \
  P2P_DRY_RUN=1 \
  bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"
  printf 'FINALIZE_STRICT top_k=%s output=%s\n' "$TOP_K" "$OUTPUT_ROOT"
  printf 'RENDER_FINAL dataset=all output=%s\n' "$OUTPUT_ROOT"
  exit 0
fi

if [[ ! -s "$PRE_FPS_ROOT/selection.json" || "$FORCE" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/select_visibility_qualitative_cases.py" \
    --result-root "$RESULT_ROOT" \
    --in-silico-root "$SILICO_ROOT" \
    --in-vitro-root "$VITRO_ROOT" \
    --output-root "$PRE_FPS_ROOT" \
    --candidate-count "$CANDIDATE_COUNT" \
    --per-metric "$PER_METRIC"
fi

if [[ ! -s "$OUTPUT_ROOT/selection.json" || "$FORCE" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/prepare_visibility_fps.py" \
    --selection "$PRE_FPS_ROOT/selection.json" \
    --output-root "$OUTPUT_ROOT" \
    --count "$POINT_COUNT"
fi

P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
P2P_PREPARED_SELECTION=1 \
bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"

"$PYTHON" "$PROJECT_ROOT/tools/select_visibility_qualitative_cases.py" \
  --finalize-root "$OUTPUT_ROOT" --top-k "$TOP_K"

"$PYTHON" "$PROJECT_ROOT/visualization/correspondence_method_comparison.py" \
  --selection "$OUTPUT_ROOT/final_selection.json" \
  --output-root "$OUTPUT_ROOT" \
  --dataset all \
  --dpi 300

printf 'FINALIZE_STRICT top_k=%s output=%s\n' "$TOP_K" "$OUTPUT_ROOT"
printf 'RENDER_FINAL dataset=all output=%s\n' "$OUTPUT_ROOT"
