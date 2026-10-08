#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
PYTHON="/home/yangx/miniconda3/envs/geo_v2/bin/python"
SILICO_ROOT="/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4"
VITRO_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
OUTPUT_ROOT="${P2P_REPRESENTATIVE_ROOT:-$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_correspondence_representative_median_2mm}"

cd "$PROJECT_ROOT"

if [[ "${P2P_FORCE_SELECTION:-0}" == "1" || ! -s "$OUTPUT_ROOT/selection.json" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/select_representative_median_correspondence.py" \
    --in-silico-root "$SILICO_ROOT" \
    --in-vitro-root "$VITRO_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --seed "${P2P_SELECTION_SEED:-20261007}"
fi

P2P_CORRESPONDENCE_ROOT="$OUTPUT_ROOT" \
P2P_PREPARED_SELECTION=1 \
P2P_DATASETS="in_silico in_vitro" \
P2P_GPU="${P2P_GPU:-1}" \
P2P_FORCE="${P2P_FORCE:-0}" \
bash "$PROJECT_ROOT/scripts/run_visibility_best3_correspondences.sh"

if [[ "${P2P_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"$PYTHON" "$PROJECT_ROOT/visualization/correspondence_method_comparison.py" \
  --selection "$OUTPUT_ROOT/selection.json" \
  --output-root "$OUTPUT_ROOT" \
  --dataset all \
  --dpi 300 \
  --panel-zoom 1.08 \
  --color-mode uniform \
  --threshold-mm 2

printf 'In-silico PNG: %s\n' "$OUTPUT_ROOT/in_silico_best3_correspondence_comparison.png"
printf 'In-vitro PNG: %s\n' "$OUTPUT_ROOT/in_vitro_best3_correspondence_comparison.png"
printf 'Metrics: %s\n' "$OUTPUT_ROOT/correspondence_metrics.csv"
