# One Correspondence Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one publication-ready matplotlib 3D figure from a real RTOR+A3 inference result, containing only the source cloud, target cloud, and GT-verified correct predicted correspondences.

**Architecture:** A standalone script loads one canonical 3D-IRCADb record and the existing cooperative RTOR+A3 checkpoint, runs the real model once, restores final fine-correspondence coordinates and the estimated transform from normalized network space to millimetres, then computes the inlier mask from the GT transform before adding a plotting-only target offset. Small pure functions isolate restoration, GT classification, confidence sampling, metrics, and rendering for tests.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Matplotlib 3D, Pillow/Poppler for artifact QA, pytest.

**Spec:** User-provided requirements in the 2026-09-30 conversation turn.

## Global Constraints

- Produce exactly one `correct_only` figure, not a multi-method comparison.
- Use the real Ours model output and checkpoint `output/geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar`.
- Default sample is the existing representative `3Dircadb1.19_vis20` from the canonical 3D-IRCADb benchmark.
- Classify correctness in original millimetre coordinates with the GT source-to-target transform and strict threshold `d < 5.0` mm.
- Apply target offset only after classification and only to plotted target coordinates/endpoints.
- Render 50–100 highest-confidence correct correspondences with red source, blue target, green lines, white background, no title, axes, or grid.
- Write PNG, PDF, and JSON to `results/correspondence_visualization/`.
- Preserve all unrelated dirty-worktree changes and do not create commits.

## Review Focus

- A plotting offset must never change the inlier mask; test identical masks regardless of offset-like coordinate values supplied only to rendering.
- Coordinate restoration must use source and target centres separately; test hand-derived millimetre coordinates.
- GT, not the predicted transform, must determine correctness; test transforms that deliberately disagree.
- Confidence selection must be deterministic and descending; test a literal score vector and the 50–100 cap.
- Empty or fewer-than-50 correct predictions must fail with a clear error rather than silently create a misleading figure.

---

### Task 1: Pure correspondence and coordinate contracts

**Files:**
- Create: `tests/test_visualize_one_correspondence.py`
- Create: `experiments/geotransformer.p2p_liver/visualization/visualize_one_correspondence.py`

**Interfaces:**
- Produces: `restore_points_mm(points, center, scale)`, `correct_correspondence_mask(src_mm, tgt_mm, gt_transform_mm, threshold_mm)`, and `select_correct_correspondences(mask, scores, max_lines, min_lines)`.

- [ ] Write tests with literal points proving separate-centre scale restoration, GT-only strict `< 5.0` classification, score ordering, and too-few-inliers rejection.
- [ ] Run `pytest -q tests/test_visualize_one_correspondence.py` with the GeoTransformer environment and confirm failure because the script does not exist.
- [ ] Implement only the pure functions and validation needed by those tests.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Real single-sample inference and metrics

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/visualization/visualize_one_correspondence.py`
- Modify: `tests/test_visualize_one_correspondence.py`

**Interfaces:**
- Consumes: Task 1 coordinate and classification functions.
- Produces: CLI inference using model outputs `src_corr_points`, `ref_corr_points`, `corr_scores`, `estimated_transform` and dataset field `transform`; JSON stats including sample/count/inlier/RRE/RTE/RMSE fields.

- [ ] Add a failing integration-oriented unit test for stats construction and exact required JSON keys/units.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement sample-id selection from the manifest, checkpoint metadata-compatible configuration, strict state-dict loading, real one-sample loader/inference, coordinate restoration, and `rigid_metrics_mm` metrics.
- [ ] Re-run the focused test and confirm it passes.

### Task 3: Matplotlib rendering and end-to-end artifact QA

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/visualization/visualize_one_correspondence.py`
- Modify: `tests/test_visualize_one_correspondence.py`
- Create at runtime: `results/correspondence_visualization/ours_correct_only.png`
- Create at runtime: `results/correspondence_visualization/ours_correct_only.pdf`
- Create at runtime: `results/correspondence_visualization/correspondence_stats.json`

**Interfaces:**
- Consumes: Task 2 restored arrays, selected indices, and stats.
- Produces: `render_correct_only(...)` and the three required artifacts.

- [ ] Add a failing rendering test asserting every green segment starts at the selected source correspondence and ends at the selected target correspondence plus offset, with axes disabled.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement matplotlib `Line3DCollection` rendering, deterministic point downsampling, equal 3D bounds, white background, and PNG/PDF/JSON output.
- [ ] Run the full script once on `3Dircadb1.19_vis20` and the epoch-150 checkpoint.
- [ ] Verify tests, PNG decode/dimensions, PDF metadata/renderability, JSON schema/count consistency, and visual appearance from a rendered preview.
- [ ] Run the relevant full project test suite and report any pre-existing failures without modifying unrelated code.

