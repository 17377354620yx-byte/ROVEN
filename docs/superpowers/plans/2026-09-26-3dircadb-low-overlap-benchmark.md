# 3D-IRCADb Low-Overlap Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 800-pair 3D-IRCADb low-overlap rigid-registration benchmark, run nine pretrained learning variants plus Go-ICP without training, and render publication comparisons with viridis distance heatmaps.

**Architecture:** RTORv2 owns the immutable millimetre-space dataset, shared prediction/metric contract, launchers, summaries, and visualization. Each external repository gets a thin native inference entry point that imports the shared contract, performs only its training-compatible preprocessing and model forward pass, then converts its predicted transform back to the common source-to-target millimetre coordinate frame.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, Open3D, Matplotlib, existing GeoTransformer/RTOR/CASTV2/DFAT/Lepard/PARENet/LiverMatch/Go-ICP code, Bash, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-3dircadb-low-overlap-benchmark-design.md`

## Global Constraints

- Input cases are exactly `3Dircadb1.1` through `3Dircadb1.20`, using `MESHES_VTK/liver.stl` in millimetres.
- Seed is exactly `20260822`; visibility levels are exactly `0.20` and `0.30`; each case has exactly 20 pair IDs at each visibility, for 800 samples.
- For each `(case_id, pair_id)`, the 20% crop is a strict index subset of the 30% crop and both share the same rigid transform and corresponding noise values.
- Rotation angle is sampled in `[25°, 90°]`; every translation component is sampled in `[-20, 20] mm`; target-only Gaussian noise has `σ=2 mm`.
- Ground truth and every saved estimate map source to target in the original millimetre coordinate frame.
- No 3D-IRCADb training, fine-tuning, parameter search, or checkpoint mutation is permitted.
- Primary success is `SR@5mm := RMSE < 5 mm`; strict recall is `RR@5°/5mm := RRE < 5° and RTE < 5 mm`, both with strict inequalities.
- PIR is `null` for methods without a coarse patch/node stage; IR is `null` when the method does not expose point correspondences.
- Solver failures and timeouts remain explicit failed records and may not be silently replaced by successful identity estimates.
- Visualization uses `viridis`, one shared color range per figure, registration overlays on row 1, and GT-distance heatmaps with `[min,max] mm` labels on row 2.

## Review Focus

- ASCII STL files with repeated vertices or degenerate triangles must still produce finite, deterministic surface samples; Task 1 tests degenerate-triangle rejection and repeatability.
- A 20%/30% pair must share transforms and nested noise/crop indices exactly, not just approximately; Task 1 compares arrays and indices directly.
- Independently centered network inputs can corrupt translation when converted back to millimetres; Task 2 round-trips a known transform through normalization.
- A crashed or timed-out Go-ICP child must yield a recorded failure without poisoning later samples; Task 8 tests timeout/error serialization.
- Missing prediction files or duplicate sample IDs must stop summary/visualization with an actionable error; Tasks 2, 9, and 10 test completeness and uniqueness checks.

---

### Task 1: Deterministic 3D-IRCADb Dataset Builder

**Files:**
- Create: `tools/build_3dircadb_low_overlap.py`
- Create: `tests/test_3dircadb_dataset_builder.py`
- Create: `scripts/build_3dircadb_low_overlap.sh`

**Interfaces:**
- Produces: `read_ascii_stl(path: Path) -> np.ndarray` with shape `(F,3,3)`.
- Produces: `sample_mesh_surface(triangles, count, rng) -> np.ndarray` with shape `(count,3)`.
- Produces: `make_nested_pair(source, case_id, pair_id, seed) -> dict[float, dict]`.
- Produces: dataset root containing `manifest.json`, `split.npz`, and 800 sample `.npz` files.

- [ ] **Step 1: Write deterministic geometry and nesting tests**

```python
def test_nested_pair_is_exact_and_repeatable():
    source = np.arange(3000, dtype=np.float64).reshape(-1, 3)
    first = make_nested_pair(source, case_id=3, pair_id=7, seed=20260822)
    second = make_nested_pair(source, case_id=3, pair_id=7, seed=20260822)
    assert np.array_equal(first[0.20]["crop_indices"], second[0.20]["crop_indices"])
    assert set(first[0.20]["crop_indices"]).issubset(first[0.30]["crop_indices"])
    assert np.array_equal(first[0.20]["transform"], first[0.30]["transform"])
    positions = {int(v): i for i, v in enumerate(first[0.30]["crop_indices"])}
    selected = [positions[int(v)] for v in first[0.20]["crop_indices"]]
    assert np.array_equal(first[0.20]["noise"], first[0.30]["noise"][selected])
```

Also assert sampled points are finite, degenerate STL facets are ignored, rotation determinant is `+1`, angle and translation bounds hold, and a second build in a temporary directory produces byte-equivalent numeric arrays.

- [ ] **Step 2: Run the tests and verify the new module is missing**

Run: `pytest -q tests/test_3dircadb_dataset_builder.py`

Expected: collection/import failure for `tools.build_3dircadb_low_overlap`.

- [ ] **Step 3: Implement ASCII-STL sampling and deterministic pair generation**

Use area-proportional triangle selection and square-root barycentric sampling:

```python
triangle_ids = np.searchsorted(np.cumsum(areas), rng.random(count) * areas.sum())
u = np.sqrt(rng.random(count))
v = rng.random(count)
points = ((1.0 - u)[:, None] * a
          + (u * (1.0 - v))[:, None] * b
          + (u * v)[:, None] * c)
```

Generate 10,000 source surface points per case. Derive an independent `np.random.SeedSequence([20260822, case_id, pair_id])` per base pair. Sort centered point projections with stable `np.argsort(..., kind="mergesort")`; keep exactly 3,000 and 2,000 indices. Transform the crop, generate 3,000 Gaussian noise vectors once, and select the matching prefix/subset for 20%.

- [ ] **Step 4: Implement the build CLI and immutable manifest**

CLI defaults:

```text
--input-root /mnt/data3/yangx/3Dircadb/3Dircadb1
--output-root /mnt/data3/yangx/3Dircadb/low_overlap_rigid_seed20260822
--seed 20260822 --surface-points 10000 --pairs-per-case 20
--visibilities 0.20,0.30 --noise-sigma-mm 2
--min-angle-deg 25 --max-angle-deg 90 --translation-mm 20
```

Write into a sibling staging directory, validate all files, then atomically rename only when the destination does not already contain a complete matching manifest. Refuse parameter-mismatched reuse unless `--output-root` changes; do not delete existing data.

- [ ] **Step 5: Run focused tests and a one-case build validation**

Run:

```bash
pytest -q tests/test_3dircadb_dataset_builder.py
/home/yangx/miniconda3/envs/geo_v2/bin/python tools/build_3dircadb_low_overlap.py \
  --input-root /mnt/data3/yangx/3Dircadb/3Dircadb1 \
  --output-root /tmp/3dircadb-smoke-20260822 --case-ids 1 --pairs-per-case 1
```

Expected: tests pass; smoke manifest has two samples with 2,000 and 3,000 target points.

- [ ] **Step 6: Add the shell wrapper and commit the task**

```bash
git add tools/build_3dircadb_low_overlap.py tests/test_3dircadb_dataset_builder.py scripts/build_3dircadb_low_overlap.sh
git commit -m "feat: build deterministic 3dircadb low-overlap pairs"
```

### Task 2: Shared Dataset, Transform, Prediction, and Metric Contract

**Files:**
- Create: `tools/ircadb_benchmark.py`
- Create: `tools/summarize_3dircadb_benchmark.py`
- Create: `tests/test_ircadb_benchmark.py`
- Modify: `tools/p2p_benchmark_metrics.py`

**Interfaces:**
- Consumes: Task 1 `manifest.json` and sample `.npz` schema.
- Produces: `load_samples(root, visibility, limit, sample_ids) -> list[SampleRecord]`.
- Produces: `normalize_pair(source_mm, target_mm) -> NormalizedPair`.
- Produces: `transform_to_network(transform_mm, pair) -> np.ndarray` and `transform_to_mm(transform_network, pair) -> np.ndarray`.
- Produces: `PredictionWriter(method, output_dir, expected_ids)` with `write_success`, `write_failure`, and `finalize`.
- Produces: `rigid_metrics_mm(source_mm, target_mm, estimate_mm, gt_mm) -> dict`.

- [ ] **Step 1: Write tests for schema validation and transform round-trip**

```python
def test_independent_centering_transform_round_trip():
    pair = normalize_pair(source_mm, target_mm)
    network_gt = transform_to_network(gt_mm, pair)
    recovered = transform_to_mm(network_gt, pair)
    np.testing.assert_allclose(recovered, gt_mm, atol=1e-8)

def test_strict_thresholds_are_not_inclusive():
    assert strict_success_metrics(5.0, 0.0, 0.0)["SR_5mm"] == 0.0
    assert strict_success_metrics(0.0, 5.0, 0.0)["RR_5deg_5mm"] == 0.0
```

Test malformed matrices, duplicate sample IDs, missing predictions, non-finite estimates, and explicit failure records.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q tests/test_ircadb_benchmark.py tests/test_p2p_benchmark_metrics.py`

Expected: import failure for the new shared module.

- [ ] **Step 3: Implement physical/network coordinate conversion**

For `x_n=(x-c_s)/s` and `y_n=(y-c_t)/s`, use:

```python
t_network = (rotation @ source_center + t_mm - target_center) / scale
t_mm = scale * t_network + target_center - rotation @ source_center
```

The complete source radius is the shared scale. Store both centers and scale in `NormalizedPair`; reject zero-radius or empty point clouds.

- [ ] **Step 4: Implement prediction files and centralized metrics**

Save one compressed `.npz` per sample with `estimated_transform`, `gt_transform`, IDs, visibility, runtime, and status. `finalize()` writes `summary.json` and `samples.csv`; it must verify every expected sample has exactly one success/failure record. Compute RRE, RTE, full-source displacement RMSE, `SR_5mm`, `RR_5deg_5mm`, Euler component errors, translation component errors, and Chamfer distance in millimetres. Set PIR/IR to `null` unless supplied by the adapter.

- [ ] **Step 5: Implement cross-method summary CLI**

`tools/summarize_3dircadb_benchmark.py` must verify manifest identity, report overall plus 20%, 30%, and per-case aggregates, and emit `summary.csv`, `summary_by_case.csv`, and `summary.json`. Missing/duplicate IDs are fatal unless `--allow-incomplete` is explicitly passed.

- [ ] **Step 6: Run tests and commit**

```bash
pytest -q tests/test_ircadb_benchmark.py tests/test_p2p_benchmark_metrics.py
git add tools/ircadb_benchmark.py tools/summarize_3dircadb_benchmark.py tools/p2p_benchmark_metrics.py tests/test_ircadb_benchmark.py
git commit -m "feat: add shared 3dircadb prediction and metric contract"
```

### Task 3: GeoTransformer-Family Adapters (Ours, GeoTransformer, DFAT)

**Files:**
- Create: `experiments/geotransformer.p2p_liver/ircadb_dataset.py`
- Create: `experiments/geotransformer.p2p_liver/test_3dircadb.py`
- Create: `tests/test_ircadb_geotransformer_adapter.py`
- Create: `/home/yangx/code/new_deform/GeoTransformer_base/experiments/geotransformer.p2p_liver/ircadb_dataset.py`
- Create: `/home/yangx/code/new_deform/GeoTransformer_base/experiments/geotransformer.p2p_liver/test_3dircadb.py`
- Create: `/home/yangx/code/Luo/DFAT-main/experiments/geotransformer.p2p_liver/ircadb_dataset.py`
- Create: `/home/yangx/code/Luo/DFAT-main/experiments/geotransformer.p2p_liver/test_3dircadb.py`

**Interfaces:**
- Consumes: Task 2 `load_samples`, normalization functions, and `PredictionWriter` through `P2P_METRICS_ROOT`.
- Produces: stack-mode dataset dictionaries with `ref_points`, `src_points`, `transform`, features, lengths, overlap, sample ID, and normalization metadata.
- Produces: millimetre-space prediction directories for `ours`, `geotransformer`, and `dfat`.

- [ ] **Step 1: Write adapter contract tests**

```python
def test_adapter_uses_target_as_reference_and_source_as_source(sample_file):
    item = IRCADbStackDataset(root=sample_file.parent, limit=1)[0]
    assert item["sample_id"] == "case01_pair00_vis020"
    recovered = transform_to_mm(item["transform"], item["normalization"])
    np.testing.assert_allclose(recovered, sample_file.gt_transform, atol=1e-5)
```

Also test `limit`, visibility filtering, deterministic sample order, and physical recovery of a synthetic model estimate.

- [ ] **Step 2: Run the contract test and verify failure**

Run: `pytest -q tests/test_ircadb_geotransformer_adapter.py`

Expected: missing adapter import.

- [ ] **Step 3: Implement the RTORv2 native dataset and tester**

Reuse `registration_collate_fn_stack_mode`, the existing backbone neighbor calibration path, `create_model(cfg)`, and strict checkpoint loading from `test.py`. Do not invoke the training loader. In `after_test_step`, convert `output['estimated_transform']` back to millimetres before calling `PredictionWriter.write_success`.

- [ ] **Step 4: Port the same contract to GeoTransformer_base**

Use its `standard_dataset.py` voxel/downsample conventions and `test_standard.py` model lifecycle. Preserve the released checkpoint configuration and write method name `geotransformer`.

- [ ] **Step 5: Port the same contract to DFAT-main**

Use DFAT's existing `create_model`, evaluator-independent forward call, and stack-mode collator. Preserve its native checkpoint load and write method name `dfat`.

- [ ] **Step 6: Run CPU dataset tests and CLI help checks**

```bash
pytest -q tests/test_ircadb_geotransformer_adapter.py
/home/yangx/miniconda3/envs/geo_v2/bin/python experiments/geotransformer.p2p_liver/test_3dircadb.py --help
/home/yangx/miniconda3/envs/geotransformer/bin/python /home/yangx/code/new_deform/GeoTransformer_base/experiments/geotransformer.p2p_liver/test_3dircadb.py --help
/home/yangx/miniconda3/envs/dfat_both/bin/python /home/yangx/code/Luo/DFAT-main/experiments/geotransformer.p2p_liver/test_3dircadb.py --help
```

Expected: tests pass and every CLI exposes `--data-root --snapshot --output --visibility --limit --save-predictions`.

- [ ] **Step 7: Run one-sample GPU smoke tests and commit only owned files in each repository**

Run all three with `--limit 1`; validate each emitted transform using Task 2. Commit RTORv2, GeoTransformer_base, and DFAT changes separately with message `feat: add 3dircadb inference adapter`.

### Task 4: CASTV2 Adapter

**Files:**
- Create: `/home/yangx/code/Luo/CASTv2/data/ircadb_data.py`
- Create: `/home/yangx/code/Luo/CASTv2/test_3dircadb.py`
- Create: `/home/yangx/code/Luo/CASTv2/tests/test_ircadb_data.py`

**Interfaces:**
- Consumes: Task 2 common sample and normalization contract.
- Produces: CAST tensors ordered as `(ref_points=target, src_points=source)` and millimetre predictions under method `castv2`.

- [ ] **Step 1: Write a dataset/transform test**

```python
def test_cast_item_and_transform_recovery(sample_root):
    item = IRCADbCASTDataset(sample_root, visibility="all", limit=1)[0]
    assert item["ref_points"].shape[-1] == 3
    assert item["src_points"].shape[-1] == 3
    np.testing.assert_allclose(
        item.denormalize(item["transform"]), item["gt_transform_mm"], atol=1e-5
    )
```

- [ ] **Step 2: Run the test and verify failure**

Run in `overpredator_py310`: `pytest -q tests/test_ircadb_data.py`.

- [ ] **Step 3: Implement the CAST dataset and inference loop**

Reuse config loading, `CAST(cfg.model)`, checkpoint handling, and `output['refined_transform']` from `test_p2p_liver.py`. Calculate PIR/IR only from actual CAST coarse/point correspondences; never synthesize them from pose success.

- [ ] **Step 4: Run CLI help, one-sample smoke test, validate output, and commit**

```bash
/home/yangx/miniconda3/envs/overpredator_py310/bin/python test_3dircadb.py --help
git add data/ircadb_data.py test_3dircadb.py tests/test_ircadb_data.py
git commit -m "feat: add 3dircadb inference adapter"
```

### Task 5: Lepard and Lepard+P2P Adapter

**Files:**
- Create: `/home/yangx/code/new_deform/Lepard/datasets/_3dircadb.py`
- Create: `/home/yangx/code/new_deform/Lepard/evaluate_3dircadb.py`
- Create: `/home/yangx/code/new_deform/Lepard/tests/test_evaluate_3dircadb.py`

**Interfaces:**
- Consumes: Task 2 contract and existing `Pipeline`, `Matching`, RANSAC, and `models.p2p.register_p2p` APIs.
- Produces: method `lepard` for `--solver ransac` and `lepard_p2p` for `--solver p2p`.

- [ ] **Step 1: Write parser, normalization, and solver-selection tests**

```python
def test_solver_names_map_to_distinct_methods():
    assert method_name("ransac") == "lepard"
    assert method_name("p2p") == "lepard_p2p"
```

Test that PIR remains `None`, IR uses exposed point matches, RANSAC fallback is recorded, and P2P fallback diagnostics are serialized.

- [ ] **Step 2: Run the tests and verify failure**

Run in `livermatch5090`: `pytest -q tests/test_evaluate_3dircadb.py`.

- [ ] **Step 3: Implement the dataset and evaluator**

Reuse `evaluate_p2p.py` model/config/checkpoint lifecycle. Use identical feature extraction for both variants and vary only the final solver. Convert `soft`, RANSAC, and P2P transforms back to millimetres before persistence; save the chosen transform plus fallback metadata.

- [ ] **Step 4: Run both one-sample modes and commit**

```bash
/home/yangx/miniconda3/envs/livermatch5090/bin/python evaluate_3dircadb.py --help
git add datasets/_3dircadb.py evaluate_3dircadb.py tests/test_evaluate_3dircadb.py
git commit -m "feat: evaluate lepard variants on 3dircadb"
```

### Task 6: PARENet Adapter

**Files:**
- Create: `/home/yangx/code/point_clould/PARENet/experiments/P2P/ircadb_dataset.py`
- Create: `/home/yangx/code/point_clould/PARENet/experiments/P2P/test_3dircadb.py`
- Create: `/home/yangx/code/point_clould/PARENet/tests/test_ircadb_adapter.py`

**Interfaces:**
- Consumes: Task 2 contract and PARENet `standard_test_data_loader` patterns.
- Produces: millimetre predictions under method `parenet`.

- [ ] **Step 1: Write a dataset and recovery test**

Test PARENet's evaluation voxel size, fixed neighbor limits, source/reference ordering, stable IDs, and normalized-to-millimetre transform recovery.

- [ ] **Step 2: Run the test and verify failure**

Run in `pare5090`: `pytest -q tests/test_ircadb_adapter.py`.

- [ ] **Step 3: Implement the dataset and tester**

Reuse strict checkpoint loading and `create_model(cfg)` from `test_standard.py`; use `cfg.data.evaluation_voxel_size` and fixed `cfg.backbone.num_neighbors`. Save coarse PIR and point IR only when supplied by native outputs.

- [ ] **Step 4: Run CLI help, one-sample GPU smoke, validate, and commit**

```bash
/home/yangx/miniconda3/envs/pare5090/bin/python experiments/P2P/test_3dircadb.py --help
git add experiments/P2P/ircadb_dataset.py experiments/P2P/test_3dircadb.py tests/test_ircadb_adapter.py
git commit -m "feat: add 3dircadb inference adapter"
```

### Task 7: LiverMatch and LiverMatch+P2P Adapter

**Files:**
- Create: `tools/evaluate_livermatch_3dircadb.py`
- Create: `tests/test_livermatch_3dircadb_adapter.py`

**Interfaces:**
- Consumes: Task 2 contract and `/home/yangx/code/point_clould/LiverMatch` legacy modules plus `p2p_util.py`.
- Produces: method `livermatch` for `--mode base` and `livermatch_p2p` for `--mode p2p`.

- [ ] **Step 1: Write legacy-import, transform, and mode tests**

```python
def test_mode_to_method():
    assert method_name("base") == "livermatch"
    assert method_name("p2p") == "livermatch_p2p"
```

Mock the legacy network output and assert both modes persist physical source-to-target transforms, distinct solver metadata, and no fabricated PIR.

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_livermatch_3dircadb_adapter.py`.

- [ ] **Step 3: Implement the two-mode evaluator**

Reuse strict checkpoint loading and the `lib.utils` compatibility injection from `evaluate_livermatch_visibility.py`. Feed the canonical sample through the project's native KPFCNN normalization/features; use `K=5` for P2P. Keep base and P2P model construction exactly aligned with their released evaluation scripts.

- [ ] **Step 4: Run both one-sample modes and commit**

```bash
/home/yangx/miniconda3/envs/livermatch5090/bin/python tools/evaluate_livermatch_3dircadb.py --help
git add tools/evaluate_livermatch_3dircadb.py tests/test_livermatch_3dircadb_adapter.py
git commit -m "feat: evaluate livermatch variants on 3dircadb"
```

### Task 8: Go-ICP Traditional Baseline

**Files:**
- Create: `/home/yangx/code/new_deform/go-icp_cython/evaluate_3dircadb.py`
- Create: `/home/yangx/code/new_deform/go-icp_cython/tests/test_evaluate_3dircadb.py`
- Modify: `/home/yangx/code/new_deform/go-icp_cython/evaluate_p2p_visibility.py`

**Interfaces:**
- Consumes: Task 2 canonical sample loader and writer.
- Produces: method `goicp` with explicit `ok`, `timeout`, or `error` per sample.

- [ ] **Step 1: Write deterministic sampling, timeout, and recovery tests**

```python
def test_timeout_is_failure_not_identity_success(monkeypatch):
    result = serialize_solver_result({"status": "timeout", "error": "Exceeded 1 seconds"})
    assert result["status"] == "timeout"
    assert result["estimated_transform"] is None
```

Also reproduce the prior spawn path and assert `_goicp_worker` is module-level/importable, deterministic subsampling uses the sample ID seed, and cube-normalized transforms recover correctly to millimetres.

- [ ] **Step 2: Run the tests and verify failure**

Run in `goicp_py310`: `pytest -q tests/test_evaluate_3dircadb.py`.

- [ ] **Step 3: Extract reusable Go-ICP worker code and implement evaluator**

Keep `_goicp_worker` at module scope. Invoke it with `multiprocessing.get_context('spawn')`. Use target as Go-ICP model and source as data, maximum 1000 points per cloud, trim 0.7, DT size 100, factor 2.0, MSE threshold 0.001, and timeout 120 seconds. On failure call `PredictionWriter.write_failure` with no estimated transform.

- [ ] **Step 4: Run spawn regression, CLI help, one-sample smoke, and commit**

```bash
/home/yangx/miniconda3/envs/goicp_py310/bin/python -m pytest -q tests/test_evaluate_3dircadb.py
/home/yangx/miniconda3/envs/goicp_py310/bin/python evaluate_3dircadb.py --help
git add evaluate_p2p_visibility.py evaluate_3dircadb.py tests/test_evaluate_3dircadb.py
git commit -m "feat: add robust 3dircadb go-icp benchmark"
```

### Task 9: Unified Launcher and Complete Result Validation

**Files:**
- Create: `scripts/run_3dircadb_benchmark.sh`
- Create: `tests/test_run_3dircadb_benchmark.py`
- Modify: `tools/summarize_3dircadb_benchmark.py`

**Interfaces:**
- Consumes: Tasks 3–8 CLIs and Task 2 summary contract.
- Produces: one command per method plus `all`, visibility selection `020|030|all`, dry-run mode, and final aggregate tables.

- [ ] **Step 1: Write launcher dry-run tests**

Parameterize all ten method keys and assert each command contains the exact environment Python, checkpoint, data root, result root, and method-specific mode/solver. Assert `all` expands each method exactly once and invalid keys exit 2.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest -q tests/test_run_3dircadb_benchmark.py`.

- [ ] **Step 3: Implement the launcher**

Expose:

```text
ours geotransformer castv2 dfat lepard lepard_p2p parenet
livermatch livermatch_p2p goicp all
```

Use exact Python paths from the design discussion, `IRCADB_GPU` (default `1`), `IRCADB_LIMIT` (default `0`), `IRCADB_DATA_ROOT`, `IRCADB_RESULT_ROOT`, and `IRCADB_DRY_RUN`. Run methods sequentially. Invoke the summary only after a complete non-limited `all all` run.

- [ ] **Step 4: Validate every dry-run expansion and commit**

```bash
IRCADB_DRY_RUN=1 ./scripts/run_3dircadb_benchmark.sh all all
pytest -q tests/test_run_3dircadb_benchmark.py
git add scripts/run_3dircadb_benchmark.sh tools/summarize_3dircadb_benchmark.py tests/test_run_3dircadb_benchmark.py
git commit -m "feat: orchestrate 3dircadb benchmark methods"
```

### Task 10: Twelve-Column Publication Visualization

**Files:**
- Create: `visualization/ircadb_method_comparison.py`
- Create: `scripts/render_3dircadb_comparison.sh`
- Create: `tests/test_ircadb_method_comparison.py`
- Modify: `visualization/paper_method_comparison.py`

**Interfaces:**
- Consumes: Task 1 canonical samples and Task 2 ten method prediction directories.
- Produces: two-row, twelve-column PNG/PDF figures, per-figure JSON statistics, and `comparison_manifest.json`.

- [ ] **Step 1: Refactor reusable rendering primitives under tests**

Move only dataset-independent helpers from `paper_method_comparison.py` into callable functions while keeping the existing P2P CLI behavior unchanged. Test explicit `--colormap`, label count, shared normalization, and transform ordering.

- [ ] **Step 2: Write missing/failed prediction and error-range tests**

```python
def test_complete_figure_requires_ten_predictions():
    with pytest.raises(ValueError, match="Missing prediction"):
        load_case_transforms(sample_id, incomplete_result_root)

def test_error_range_is_in_physical_mm():
    error = pointwise_gt_error_mm(source_mm, estimate_mm, gt_mm)
    np.testing.assert_allclose(error, expected_mm)
```

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest -q tests/test_ircadb_method_comparison.py tests/test_error_heatmap.py`.

- [ ] **Step 4: Implement the 3D-IRCADb visualization CLI**

Columns are `Initial`, the ten methods in approved order, and `Ground Truth`. Render registered source plus noisy target on row 1; render pointwise distance from estimated transformed source to GT transformed source on row 2. Default `--colormap viridis`; annotate every heatmap with `[min,max] mm`; derive one p95-clipped scale from registered methods for each figure and record the exact range in JSON.

- [ ] **Step 5: Add filters and wrapper**

Support `--case-ids`, `--pair-ids`, `--visibility 0.20|0.30|all`, `--sample-ids`, `--dpi`, `--colormap`, and `--allow-failed`. Failed methods are labeled `FAILED` only when `--allow-failed` is explicitly used.

- [ ] **Step 6: Render a synthetic test figure and commit**

```bash
pytest -q tests/test_ircadb_method_comparison.py tests/test_error_heatmap.py
git add visualization/paper_method_comparison.py visualization/ircadb_method_comparison.py scripts/render_3dircadb_comparison.sh tests/test_ircadb_method_comparison.py
git commit -m "feat: render 3dircadb method comparison figures"
```

### Task 11: End-to-End Verification and Operator Commands

**Files:**
- Create: `docs/3DIRCADB_LOW_OVERLAP_BENCHMARK.md`
- Modify: `docs/superpowers/specs/2026-09-26-3dircadb-low-overlap-benchmark-design.md`

**Interfaces:**
- Consumes: every previous task.
- Produces: verified commands for dataset generation, each of ten methods, aggregation, and visualization.

- [ ] **Step 1: Run all RTORv2 unit tests added by this plan**

```bash
pytest -q \
  tests/test_3dircadb_dataset_builder.py \
  tests/test_ircadb_benchmark.py \
  tests/test_ircadb_geotransformer_adapter.py \
  tests/test_livermatch_3dircadb_adapter.py \
  tests/test_run_3dircadb_benchmark.py \
  tests/test_ircadb_method_comparison.py \
  tests/test_p2p_benchmark_metrics.py \
  tests/test_error_heatmap.py
```

Expected: all pass.

- [ ] **Step 2: Run external-repository focused tests in their assigned environments**

Run CASTV2, Lepard, PARENet, and Go-ICP tests listed in Tasks 4–8. Expected: all pass.

- [ ] **Step 3: Build the full dataset and validate its manifest**

```bash
./scripts/build_3dircadb_low_overlap.sh
/home/yangx/miniconda3/envs/geo_v2/bin/python tools/build_3dircadb_low_overlap.py \
  --validate-only \
  --output-root /mnt/data3/yangx/3Dircadb/low_overlap_rigid_seed20260822
```

Expected: 800 valid samples, 400 per visibility, 40 per case, and 20 nested pair identities per case.

- [ ] **Step 4: Run one sample through all ten methods**

```bash
IRCADB_LIMIT=1 ./scripts/run_3dircadb_benchmark.sh all all
```

Expected: ten method summaries, no schema/coordinate errors, and Go-ICP either succeeds or records an explicit solver failure without crashing the launcher.

- [ ] **Step 5: Generate one comparison figure from the smoke outputs**

Run the visualization with `--sample-ids` set to the smoke ID, `--colormap viridis`, and `--allow-failed` only if Go-ICP explicitly timed out. Verify PNG dimensions, twelve titles, two rows, the color bar, and `[min,max] mm` annotations.

- [ ] **Step 6: Write the operator document and mark the design implemented**

Document exact per-model and all-model commands, environment variables, output paths, expected runtime caveat for 800 Go-ICP samples, resume behavior, completeness validation, aggregation, and visualization. Change the design status to `Implemented and verified` only after Steps 1–5 succeed.

- [ ] **Step 7: Check repository cleanliness for owned paths and commit documentation**

```bash
git diff --check
git status --short
git add docs/3DIRCADB_LOW_OVERLAP_BENCHMARK.md docs/superpowers/specs/2026-09-26-3dircadb-low-overlap-benchmark-design.md
git commit -m "docs: add 3dircadb benchmark runbook"
```

