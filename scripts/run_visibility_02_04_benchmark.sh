#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
CONDITION="${2:-all}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 {ours|geotransformer|castv2|dfat|lepard|lepard_p2p|parenet|livermatch|livermatch_p2p|rtorv8_no_proposal|goicp|all} [in_silico_none|in_silico_noise4|in_vitro|all]" >&2
  exit 2
fi

case "$MODEL" in
  ours|geotransformer|castv2|dfat|lepard|lepard_p2p|parenet|livermatch|livermatch_p2p|rtorv8_no_proposal|goicp|all) ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
case "$CONDITION" in
  in_silico_none|in_silico_noise4|in_vitro|all) ;;
  *) echo "Unknown condition: $CONDITION" >&2; exit 2 ;;
esac

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
SILICO_ROOT="/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4"
VITRO_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
RESULT_ROOT="${P2P_BENCHMARK_RESULTS:-$PROJECT_ROOT/results/visibility_0.2_0.4_benchmark_strict_v3}"
GPU="${P2P_GPU:-1}"
TEST_LIMIT="${P2P_BENCHMARK_LIMIT:-0}"

OURS_ROOT="$PROJECT_ROOT"
GEO_ROOT="/home/yangx/code/new_deform/GeoTransformer_base"
CAST_ROOT="/home/yangx/code/Luo/CASTv2"
DFAT_ROOT="/home/yangx/code/Luo/DFAT-main"
LEPARD_ROOT="/home/yangx/code/new_deform/Lepard"
PARE_ROOT="/home/yangx/code/point_clould/PARENet"
RTORV8_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv8"
GOICP_ROOT="/home/yangx/code/new_deform/go-icp_cython"

OURS_CKPT="$OURS_ROOT/output/geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar"
GEO_CKPT="$GEO_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
CAST_CKPT="$CAST_ROOT/ckpt/p2p_liver/castv2-p2p-liver-epoch-150.pth.tar"
DFAT_CKPT="$DFAT_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
LEPARD_CKPT="$LEPARD_ROOT/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth"
PARE_CKPT="$PARE_ROOT/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar"
LIVERMATCH_CKPT="/home/yangx/code/point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"
RTORV8_CKPT="$RTORV8_ROOT/output/geotransformer.p2p_liver.rtor_a3_cooperative_abl1_no_proposal_seed7351/snapshots/epoch-150.pth.tar"

PY_GEO_V2="/home/yangx/miniconda3/envs/geo_v2/bin/python"
PY_GEOTRANSFORMER="/home/yangx/miniconda3/envs/geotransformer/bin/python"
PY_CAST="/home/yangx/miniconda3/envs/overpredator_py310/bin/python"
PY_DFAT="/home/yangx/miniconda3/envs/dfat_both/bin/python"
PY_LIVERMATCH="/home/yangx/miniconda3/envs/livermatch5090/bin/python"
PY_PARE="/home/yangx/miniconda3/envs/pare5090/bin/python"
PY_RTORV8="/home/yangx/miniconda3/envs/geo_py310/bin/python"
PY_GOICP="/home/yangx/miniconda3/envs/goicp_py310/bin/python"

mkdir -p "$RESULT_ROOT"

resolve_condition() {
  case "$1" in
    in_silico_none)
      DATASET="in_silico"; NOISE="none"; DATA_ROOT="$SILICO_ROOT" ;;
    in_silico_noise4)
      DATASET="in_silico"; NOISE="4"; DATA_ROOT="$SILICO_ROOT" ;;
    in_vitro)
      DATASET="in_vitro"; NOISE="none"; DATA_ROOT="$VITRO_ROOT" ;;
  esac
}

run_logged() {
  local output="$1"
  shift
  mkdir -p "$(dirname "$output")"
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${P2P_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee "${output%.json}.log"
}

common_env() {
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
    P2P_TRAIN_DATA_ROOT="/mnt/data3/yangx/P2P/Dataset" \
    P2P_METRICS_ROOT="$PROJECT_ROOT" \
    P2P_DATA_ROOT="$SILICO_ROOT" \
    P2P_IN_VITRO_ROOT="$VITRO_ROOT" \
    "$@"
}

run_ours() {
  local output="$RESULT_ROOT/ours/$1.json"
  run_logged "$output" common_env P2P_RUN_NAME=rtor_a3_cooperative \
    "$PY_GEO_V2" "$OURS_ROOT/experiments/geotransformer.p2p_liver/test.py" \
    --snapshot "$OURS_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --architecture rtor_a3 --interaction_profile cooperative \
    --test_limit "$TEST_LIMIT" --output "$output"
}

run_geotransformer() {
  local output="$RESULT_ROOT/geotransformer/$1.json"
  run_logged "$output" common_env \
    "$PY_GEOTRANSFORMER" "$GEO_ROOT/experiments/geotransformer.p2p_liver/test_standard.py" \
    --snapshot "$GEO_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --test_limit "$TEST_LIMIT" --output "$output"
}

run_castv2() {
  local output="$RESULT_ROOT/castv2/$1.json"
  run_logged "$output" common_env \
    "$PY_CAST" "$CAST_ROOT/test_p2p_liver.py" \
    --config "$CAST_ROOT/config/p2p_liver.json" --snapshot "$CAST_CKPT" \
    --dataset "$DATASET" --noise "$NOISE" --test_limit "$TEST_LIMIT" --output "$output"
}

run_dfat() {
  local output="$RESULT_ROOT/dfat/$1.json"
  run_logged "$output" common_env \
    "$PY_DFAT" "$DFAT_ROOT/experiments/geotransformer.p2p_liver/test.py" \
    --snapshot "$DFAT_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --test_limit "$TEST_LIMIT" --output "$output"
}

run_lepard_variant() {
  local method="$1"
  local condition="$2"
  local solver="ransac"
  [[ "$method" == "lepard_p2p" ]] && solver="p2p"
  local output="$RESULT_ROOT/$method/$condition.json"
  run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "$PY_LIVERMATCH" "$LEPARD_ROOT/evaluate_p2p.py" \
    --config "$LEPARD_ROOT/configs/train/p2p_liver.yaml" \
    --checkpoint "$LEPARD_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --solver "$solver" --in-silico-root "$SILICO_ROOT/Deform_mesh_npz_test/Test" \
    --in-silico-list "$SILICO_ROOT/Deform_mesh_npz_test/list.npz" \
    --in-vitro-root "$VITRO_ROOT" --in-vitro-list "$VITRO_ROOT/rigid_list.npy" \
    --limit "$TEST_LIMIT" --output "$output" --device cuda:0
}

run_parenet() {
  local output="$RESULT_ROOT/parenet/$1.json"
  local pare_torch_lib
  pare_torch_lib="$($PY_PARE -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  run_logged "$output" env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
    P2P_DATA_ROOT="$SILICO_ROOT" P2P_IN_VITRO_ROOT="$VITRO_ROOT" \
    LD_LIBRARY_PATH="$pare_torch_lib:/home/yangx/miniconda3/envs/pare5090/lib:/home/yangx/miniconda3/envs/pare5090/lib64:${LD_LIBRARY_PATH:-}" \
    "$PY_PARE" "$PARE_ROOT/experiments/P2P/test_standard.py" \
    --snapshot "$PARE_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --test_limit "$TEST_LIMIT" --num_workers 0 --output "$output"
}

run_livermatch_variant() {
  local method="$1"
  local condition="$2"
  local mode="base"
  [[ "$method" == "livermatch_p2p" ]] && mode="p2p"
  local output="$RESULT_ROOT/$method/$condition.json"
  run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true \
    "$PY_LIVERMATCH" "$PROJECT_ROOT/tools/evaluate_livermatch_visibility.py" \
    --mode "$mode" --dataset "$DATASET" --noise "$NOISE" --data-root "$DATA_ROOT" \
    --checkpoint "$LIVERMATCH_CKPT" --k 5 --limit "$TEST_LIMIT" \
    --output "$output" --device cuda:0
}

run_rtorv8_no_proposal() {
  local output="$RESULT_ROOT/rtorv8_no_proposal/$1.json"
  run_logged "$output" common_env P2P_RUN_NAME=rtor_a3_cooperative_abl1_no_proposal_seed7351 \
    "$PY_RTORV8" "$RTORV8_ROOT/experiments/geotransformer.p2p_liver/test.py" \
    --snapshot "$RTORV8_CKPT" --dataset "$DATASET" --noise "$NOISE" \
    --architecture rtor_a3 --interaction_profile cooperative \
    --ablation_profile abl1_no_proposal \
    --test_limit "$TEST_LIMIT" --output "$output"
}

run_goicp() {
  local output="$RESULT_ROOT/goicp/$1.json"
  run_logged "$output" env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
    P2P_METRICS_ROOT="$PROJECT_ROOT" \
    "$PY_GOICP" "$GOICP_ROOT/evaluate_p2p_visibility.py" \
    --dataset "$DATASET" --noise "$NOISE" --data-root "$DATA_ROOT" \
    --goicp-root "$GOICP_ROOT" --max-points 1000 --trim-fraction 0.7 \
    --dt-size 100 --mse-threshold 0.001 --timeout 120 \
    --limit "$TEST_LIMIT" --output "$output"
}

run_one() {
  local method="$1"
  local condition="$2"
  resolve_condition "$condition"
  case "$method" in
    ours) run_ours "$condition" ;;
    geotransformer) run_geotransformer "$condition" ;;
    castv2) run_castv2 "$condition" ;;
    dfat) run_dfat "$condition" ;;
    lepard|lepard_p2p) run_lepard_variant "$method" "$condition" ;;
    parenet) run_parenet "$condition" ;;
    livermatch|livermatch_p2p) run_livermatch_variant "$method" "$condition" ;;
    rtorv8_no_proposal) run_rtorv8_no_proposal "$condition" ;;
    goicp) run_goicp "$condition" ;;
  esac
}

MODELS=(ours geotransformer castv2 dfat lepard lepard_p2p parenet livermatch livermatch_p2p rtorv8_no_proposal goicp)
CONDITIONS=(in_silico_none in_silico_noise4 in_vitro)
[[ "$MODEL" != "all" ]] && MODELS=("$MODEL")
[[ "$CONDITION" != "all" ]] && CONDITIONS=("$CONDITION")

for method in "${MODELS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    run_one "$method" "$condition"
  done
done

if [[ "${P2P_DRY_RUN:-0}" != "1" && "$TEST_LIMIT" == "0" ]]; then
  "$PY_GEO_V2" "$PROJECT_ROOT/scripts/summarize_visibility_02_04.py" \
    --result-root "$RESULT_ROOT" \
    --output "$RESULT_ROOT/summary.csv"
fi
