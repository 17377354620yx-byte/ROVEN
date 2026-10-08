# Visibility [0.2, 0.4] benchmark

The launcher evaluates the 1,538-case in-silico subset and the 204-case
in-vitro subset. It writes per-case JSON/CSV/NPY files and an aggregate
`summary.csv`/`summary.md` under:

```text
results/visibility_0.2_0.4_benchmark_strict_v3/
```

The three conditions are:

- `in_silico_none`: stored target without additional synthetic noise.
- `in_silico_noise4`: stored target plus the official `npz["4"]` perturbation.
- `in_vitro`: real phantom data; no synthetic noise is permitted.

The primary endpoint is volumetric-marker RMS-TRE in millimetres. Primary
success is `SR@5 mm`. Strict registration recall is `RR@5deg/5mm`, requiring
both `RRE < 5 degrees` and `RTE < 5 mm`. `SR@20 mm` and the historical
`RRE < 15 degrees` plus normalized `RTE < 0.3` recall are retained only as
legacy fields. P2P variants
use `K=5`. The extended registration protocol is:

- `PIR`: precision of predicted coarse patch/node pairs. It is `N/A` for
  Lepard, LiverMatch, and Go-ICP because these models do not expose a separate
  coarse correspondence stage.
- `IR`: fraction of predicted point correspondences with residual below 0.1
  in normalized coordinates under the fitted rigid oracle. It is `N/A` for
  Go-ICP because the solver does not expose predicted correspondences.
- `RRE` and `RTE`: isotropic rotation error in degrees and translation error
  in millimetres.
- `RMSE`: true root-mean-square displacement, in millimetres, between source
  points transformed by the prediction and by the fitted rigid oracle.
- `RR`: strict `RRE < 5 degrees` and `RTE < 5 mm`.
- `r_rmse`/`r_mae`: wrapped XYZ Euler component errors in degrees.
- `t_rmse`/`t_mae`: XYZ translation component errors in millimetres.
- `chamfer_dist`: average bidirectional nearest-neighbour L2 distance between
  the transformed source and target, in millimetres.

The oracle is the least-squares rigid transform fitted from paired volumetric
markers. RMS-TRE/SR@5 mm are the primary deformation-aware endpoints;
SR@20 mm remains in the output because the liver data are
deformable and those marker endpoints measure errors that a rigid oracle does
not capture.

Go-ICP's project-local `evaluate_p2p_visibility.py` uses the partial target as
`model` and complete source as `data`, then
recovers the source-to-target transform. Both clouds are independently centered,
jointly scaled into `[-1, 1]^3`, and deterministically sampled to at most 1,000
points each. Parameters are trimming fraction `0.7`, DT size `100`, DT factor
`2.0`, MSE threshold `0.001`, and per-case timeout `120 s`. A timed-out or failed
case is retained in the denominator with its status recorded and an identity
fallback. For in-silico noise-4, the official saved `npz["4"]` target noise is
added before `rot_tgt`, matching the P2P test adapter.

## One command per model

Each command below runs all three conditions for one model:

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2

P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh ours all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh geotransformer all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh castv2 all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh dfat all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh lepard all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh lepard_p2p all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh parenet all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh livermatch all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh livermatch_p2p all
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh rtorv8_no_proposal all
scripts/run_visibility_02_04_benchmark.sh goicp all
```

Run all 33 evaluations sequentially:

```bash
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh all all
```

Run only one condition by replacing `all` with `in_silico_none`,
`in_silico_noise4`, or `in_vitro`, for example:

```bash
P2P_GPU=1 scripts/run_visibility_02_04_benchmark.sh ours in_silico_noise4
```

Print the expanded underlying commands without running them:

```bash
P2P_DRY_RUN=1 scripts/run_visibility_02_04_benchmark.sh all all
```

For a smoke test, set a sample limit. Limited runs deliberately skip aggregate
table generation because they are not complete benchmark results:

```bash
P2P_GPU=1 P2P_BENCHMARK_LIMIT=1 \
  P2P_BENCHMARK_RESULTS=/tmp/p2p_visibility_smoke \
  scripts/run_visibility_02_04_benchmark.sh all in_silico_noise4
```

Rebuild the aggregate table after complete runs:

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python \
  scripts/summarize_visibility_02_04.py \
  --result-root results/visibility_0.2_0.4_benchmark_strict_v3 \
  --output results/visibility_0.2_0.4_benchmark_strict_v3/summary.csv
```
