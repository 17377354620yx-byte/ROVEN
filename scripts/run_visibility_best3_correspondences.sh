#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
OLD_ROOT="$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_livermatch_style"
OUTPUT_ROOT="${P2P_CORRESPONDENCE_ROOT:-$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_correspondence_best3}"
RESULT_ROOT="$PROJECT_ROOT/results/visibility_0.2_0.4_benchmark_metrics_v2"
SILICO_ROOT="/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4"
VITRO_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
RAW_ROOT="$OUTPUT_ROOT/raw"
SELECTION="$OUTPUT_ROOT/selection.json"
GPU="${P2P_GPU:-1}"
DRY_RUN="${P2P_DRY_RUN:-0}"
FORCE="${P2P_FORCE:-0}"
PREPARED_SELECTION="${P2P_PREPARED_SELECTION:-0}"

GEO_ROOT="/home/yangx/code/new_deform/GeoTransformer_base"
CAST_ROOT="/home/yangx/code/Luo/CASTv2"
DFAT_ROOT="/home/yangx/code/Luo/DFAT-main"
LEPARD_ROOT="/home/yangx/code/new_deform/Lepard"
PREDATOR_ROOT="/home/yangx/code/OverlapPredator"
PARE_ROOT="/home/yangx/code/point_clould/PARENet"

OURS_CKPT="$PROJECT_ROOT/output/geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar"
GEO_CKPT="$GEO_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
CAST_CKPT="$CAST_ROOT/ckpt/p2p_liver/castv2-p2p-liver-epoch-150.pth.tar"
DFAT_CKPT="$DFAT_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
LEPARD_CKPT="$LEPARD_ROOT/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth"
PREDATOR_CKPT="$PREDATOR_ROOT/snapshot/p2p_liver/checkpoints/epoch_150.pth"
PARE_CKPT="$PARE_ROOT/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar"
LIVERMATCH_CKPT="/home/yangx/code/point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"

PY_GEO_V2="/home/yangx/miniconda3/envs/geo_v2/bin/python"
PY_GEO="/home/yangx/miniconda3/envs/geotransformer/bin/python"
PY_CAST="/home/yangx/miniconda3/envs/overpredator_py310/bin/python"
PY_DFAT="/home/yangx/miniconda3/envs/dfat_both/bin/python"
PY_LIVER="/home/yangx/miniconda3/envs/livermatch5090/bin/python"
PY_PARE="/home/yangx/miniconda3/envs/pare5090/bin/python"

METHODS=(ours geotransformer castv2 dfat lepard predator parenet livermatch livermatch_p2p)
read -r -a DATASETS <<< "${P2P_DATASETS:-in_silico in_vitro}"

selection_count() {
  local dataset="$1"
  "$PY_GEO_V2" - "$SELECTION" "$dataset" <<'PY'
import json
from pathlib import Path
import sys

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(selection["datasets"][sys.argv[2]]["cases"]))
PY
}

checkpoint_for() {
  case "$1" in
    ours) printf '%s\n' "$OURS_CKPT" ;;
    geotransformer) printf '%s\n' "$GEO_CKPT" ;;
    castv2) printf '%s\n' "$CAST_CKPT" ;;
    dfat) printf '%s\n' "$DFAT_CKPT" ;;
    lepard) printf '%s\n' "$LEPARD_CKPT" ;;
    predator) printf '%s\n' "$PREDATOR_CKPT" ;;
    parenet) printf '%s\n' "$PARE_CKPT" ;;
    livermatch|livermatch_p2p) printf '%s\n' "$LIVERMATCH_CKPT" ;;
  esac
}

python_for() {
  case "$1" in
    ours) printf '%s\n' "$PY_GEO_V2" ;;
    geotransformer) printf '%s\n' "$PY_GEO" ;;
    castv2|predator) printf '%s\n' "$PY_CAST" ;;
    dfat) printf '%s\n' "$PY_DFAT" ;;
    lepard|livermatch|livermatch_p2p) printf '%s\n' "$PY_LIVER" ;;
    parenet) printf '%s\n' "$PY_PARE" ;;
  esac
}

if [[ "$DRY_RUN" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      if [[ "$method" == "predator" ]]; then
        test_limit="selection_size"
        [[ -s "$SELECTION" ]] && test_limit="$(selection_count "$dataset")"
        printf 'MODEL_CMD %s %s %s checkpoint=%s --test-limit %s --save-correspondences-dir %s/raw/%s/%s_predictions output=%s\n' \
          "$method" "$dataset" "$(python_for "$method")" "$(checkpoint_for "$method")" \
          "$test_limit" "$OUTPUT_ROOT" "$dataset" "$method" "$OUTPUT_ROOT"
      else
        printf 'MODEL_CMD %s %s %s checkpoint=%s --save-correspondences-dir %s/raw/%s/%s_predictions output=%s\n' \
          "$method" "$dataset" "$(python_for "$method")" "$(checkpoint_for "$method")" \
          "$OUTPUT_ROOT" "$dataset" "$method" "$OUTPUT_ROOT"
      fi
    done
  done
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
if [[ "$PREPARED_SELECTION" != "1" && ( ! -s "$SELECTION" || "$FORCE" == "1" ) ]]; then
  "$PY_GEO_V2" "$PROJECT_ROOT/tools/prepare_visibility_best3.py" \
    --result-root "$RESULT_ROOT" \
    --in-silico-root "$SILICO_ROOT" --in-vitro-root "$VITRO_ROOT" \
    --output-root "$OUTPUT_ROOT" --top-k 3
fi

SILICO_SUBSET="$OUTPUT_ROOT/subsets/in_silico"
VITRO_SUBSET="$OUTPUT_ROOT/subsets/in_vitro"

run_logged() {
  local output="$1"
  shift
  mkdir -p "$(dirname "$output")"
  "$@" 2>&1 | tee "${output%.json}.log"
}

collect_method() {
  local method="$1" dataset="$2" raw_dir="$3"
  "$PY_GEO_V2" "$PROJECT_ROOT/tools/collect_visibility_correspondences.py" \
    --method "$method" --dataset "$dataset" --selection "$SELECTION" \
    --raw-dir "$raw_dir" --checkpoint "$(checkpoint_for "$method")" \
    --output-root "$OUTPUT_ROOT"
}

cache_complete() {
  local method="$1" dataset="$2"
  [[ "$FORCE" != "1" && -s "$OUTPUT_ROOT/correspondences/$dataset/${method}_summary.json" ]]
}

run_method() {
  local method="$1" dataset="$2" data_root output raw_dir mode
  if cache_complete "$method" "$dataset"; then
    echo "Cached: $method/$dataset"
    return
  fi
  output="$RAW_ROOT/$dataset/$method.json"
  data_root="$SILICO_SUBSET"
  [[ "$dataset" == "in_vitro" ]] && data_root="$VITRO_SUBSET"
  case "$method" in
    ours)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" \
        P2P_RUN_NAME=visibility_best3_correspondence_ours \
        P2P_DATA_ROOT="$SILICO_SUBSET" P2P_IN_VITRO_ROOT="$VITRO_SUBSET" \
        P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        "$PY_GEO_V2" "$PROJECT_ROOT/experiments/geotransformer.p2p_liver/test.py" \
        --snapshot "$OURS_CKPT" --dataset "$dataset" --noise none \
        --architecture rtor_a3 --interaction_profile cooperative --save_predictions --output "$output"
      raw_dir="$PROJECT_ROOT/output/geotransformer.p2p_liver.visibility_best3_correspondence_ours/registration"
      collect_method "$method" "$dataset" "$raw_dir"
      ;;
    geotransformer)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$SILICO_SUBSET" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_GEO" \
        "$GEO_ROOT/experiments/geotransformer.p2p_liver/test_standard.py" \
        --snapshot "$GEO_CKPT" --dataset "$dataset" --noise none --save_predictions --output "$output"
      collect_method "$method" "$dataset" "$GEO_ROOT/output/geotransformer.p2p_liver/registration"
      ;;
    castv2)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$data_root" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" P2P_METRICS_ROOT="$PROJECT_ROOT" \
        "$PY_CAST" "$CAST_ROOT/test_p2p_liver.py" --config "$CAST_ROOT/config/p2p_liver.json" \
        --snapshot "$CAST_CKPT" --dataset "$dataset" --noise none --save_predictions --output "$output"
      collect_method "$method" "$dataset" "${output%.json}_predictions"
      ;;
    dfat)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$SILICO_SUBSET" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_DFAT" \
        "$DFAT_ROOT/experiments/geotransformer.p2p_liver/test.py" \
        --snapshot "$DFAT_CKPT" --dataset "$dataset" --noise none --save_predictions --output "$output"
      collect_method "$method" "$dataset" "$DFAT_ROOT/output/geotransformer.p2p_liver/registration"
      ;;
    lepard)
      raw_dir="${output%.json}_predictions"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" "$PY_LIVER" "$LEPARD_ROOT/evaluate_p2p.py" \
        --config "$LEPARD_ROOT/configs/train/p2p_liver.yaml" --checkpoint "$LEPARD_CKPT" \
        --dataset "$dataset" --noise none --solver ransac \
        --in-silico-root "$SILICO_SUBSET/Deform_mesh_npz_test/Test" \
        --in-silico-list "$SILICO_SUBSET/Deform_mesh_npz_test/list.npz" \
        --in-vitro-root "$VITRO_SUBSET" --in-vitro-list "$VITRO_SUBSET/rigid_list.npy" \
        --save-correspondences-dir "$raw_dir" --output "$output" --device cuda:0
      collect_method "$method" "$dataset" "$raw_dir"
      ;;
    predator)
      local test_limit
      test_limit="$(selection_count "$dataset")"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$SILICO_SUBSET" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" "$PY_CAST" "$PREDATOR_ROOT/test_p2p_liver.py" \
        --config "$PREDATOR_ROOT/configs/train/p2p_liver.yaml" --snapshot "$PREDATOR_CKPT" \
        --dataset "$dataset" --data-root "$data_root" --noise none --test-limit "$test_limit" \
        --save-predictions --output "$output" --device cuda:0
      collect_method "$method" "$dataset" "${output%.json}_predictions"
      ;;
    parenet)
      local pare_torch_lib
      pare_torch_lib="$($PY_PARE -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_RUN_NAME=visibility_best3_correspondence \
        P2P_DATA_ROOT="$SILICO_SUBSET" P2P_IN_VITRO_ROOT="$VITRO_SUBSET" \
        LD_LIBRARY_PATH="$pare_torch_lib:/home/yangx/miniconda3/envs/pare5090/lib:/home/yangx/miniconda3/envs/pare5090/lib64:${LD_LIBRARY_PATH:-}" \
        "$PY_PARE" "$PARE_ROOT/experiments/P2P/test_standard.py" --snapshot "$PARE_CKPT" \
        --dataset "$dataset" --noise none --num_workers 0 --save_predictions --output "$output"
      collect_method "$method" "$dataset" "${output%.json}_predictions"
      ;;
    livermatch|livermatch_p2p)
      mode=base
      [[ "$method" == "livermatch_p2p" ]] && mode=p2p
      raw_dir="${output%.json}_predictions"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true \
        "$PY_LIVER" "$PROJECT_ROOT/tools/evaluate_livermatch_visibility.py" --mode "$mode" \
        --dataset "$dataset" --noise none --data-root "$data_root" --checkpoint "$LIVERMATCH_CKPT" \
        --k 5 --save-correspondences-dir "$raw_dir" --output "$output" --device cuda:0
      collect_method "$method" "$dataset" "$raw_dir"
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_method "$method" "$dataset"
  done
done

echo "Correspondence caches written under $OUTPUT_ROOT/correspondences"
