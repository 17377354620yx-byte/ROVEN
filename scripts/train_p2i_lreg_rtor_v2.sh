#!/usr/bin/env bash
set -euo pipefail

P2I_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2I_PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export P2I_LREG_RUN_NAME="${P2I_LREG_RUN_NAME:-rtor_a3}"

exec conda run --no-capture-output -n geo_v2 \
  python experiments/geotransformer.p2i_lreg/trainval.py \
  --architecture rtor_a3 --log_steps 10 "$@"
