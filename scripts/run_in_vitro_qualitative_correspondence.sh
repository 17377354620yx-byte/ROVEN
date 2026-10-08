#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
PYTHON="/home/yangx/miniconda3/envs/geo_v2/bin/python"
RESULT_ROOT="$PROJECT_ROOT/results/visibility_0.2_0.4_benchmark_strict_v3"
DATA_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
OUTPUT_ROOT="${P2P_QUAL_CORR_ROOT:-$PROJECT_ROOT/output/visualization/in_vitro_qualitative_correspondence}"

if [[ "${P2P_FORCE_SELECTION:-0}" == "1" || ! -s "$OUTPUT_ROOT/selection.json" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/select_in_vitro_qualitative_correspondence.py" \
    --result-root "$RESULT_ROOT" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --selection-strategy "${P2P_SELECTION_STRATEGY:-representative}"
fi

P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
P2P_PREPARED_SELECTION=1 \
P2P_DATASETS=in_vitro \
P2P_GPU="${P2P_GPU:-1}" \
P2P_FORCE="${P2P_FORCE:-0}" \
bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"

if [[ "${P2P_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"$PYTHON" "$PROJECT_ROOT/visualization/correspondence_method_comparison.py" \
  --selection "$OUTPUT_ROOT/selection.json" \
  --output-root "$OUTPUT_ROOT" \
  --dataset in_vitro \
  --dpi 300 \
  --panel-zoom 1.08

printf 'PNG: %s\n' "$OUTPUT_ROOT/in_vitro_best3_correspondence_comparison.png"
printf 'PDF: %s\n' "$OUTPUT_ROOT/in_vitro_best3_correspondence_comparison.pdf"
printf 'Metrics: %s\n' "$OUTPUT_ROOT/correspondence_metrics.csv"
