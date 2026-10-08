# P2P Correspondence Visualization Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the existing real-model correspondence visualization pipeline for one representative in-silico and one representative in-vitro sample from the two P2P visibility `[0.2, 0.4]` subsets.

**Architecture:** Keep checkpoint loading, RTOR+A3 inference, GT-distance classification, confidence selection, and rendering unchanged. Add a P2P adapter that delegates network input and oracle construction to the existing `LiverTask3TestDataset` / `LiverTask3InVitroTestDataset`, while independently recovering the exact raw millimetre clouds, per-cloud centres, shared source radius, and source-to-target marker-oracle transform needed to restore model outputs.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Matplotlib 3D, pytest.

**Spec:** User request in the 2026-09-30 conversation turn extending `visualize_one_correspondence.py` to two P2P datasets.

## Global Constraints

- Do not change the Ours model, checkpoint, correct-correspondence rule, confidence ranking, visualization offset semantics, colors, or Matplotlib rendering style.
- In-silico root is `/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4`; in-vitro root is `/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4`.
- Use the current project dataset transforms: marker-paired least-squares rigid oracle in `data['transform']`; in-silico target uses `rot_tgt` exactly as `LiverTask3TestDataset` does.
- Restore `src_corr_points` with source centre and `ref_corr_points` with target centre, both using the complete-source radius; all GT checks and metrics are in millimetres.
- Default representative samples are chosen from existing real Ours benchmark results for high IR and good registration: `00046/00002/00003.npz` and `5_00020.npz`.
- Produce one correct-only PNG, PDF, and JSON per dataset in `results/correspondence_visualization/`.
- Preserve unrelated dirty-worktree changes and do not commit.

## Review Focus

- In-silico target rotation must be applied to both surface points and markers before centre/GT recovery.
- The adapter's normalized points and transform must exactly match the existing dataset item.
- Dataset subset index, sample name, and raw file path must not be confused with `original_indices.npy`.
- In-vitro root conventions differ from in-silico and must resolve `Rigid_test_data/<sample>` correctly.
- Offset remains render-only and the threshold remains strict `d < 5.0 mm` after restoration.

---

### Task 1: P2P data and coordinate adapter

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/visualization/visualize_one_correspondence.py`
- Modify: `tests/test_visualize_one_correspondence.py`

**Interfaces:**
- Produces: `load_p2p_sample(cfg, dataset_kind, data_root, sample_id)` returning the selected existing dataset/subset plus raw millimetre context and normalization metadata.

- [x] Add failing tests against one real sample per dataset proving the adapter sample arrays and network GT exactly equal the existing dataset item.
- [x] Run focused tests and confirm failure because the adapter is absent.
- [x] Implement in-silico and in-vitro loading without changing model preprocessing.
- [x] Re-run focused tests and confirm they pass.

### Task 2: Generalized inference and dataset-specific outputs

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/visualization/visualize_one_correspondence.py`
- Modify: `tests/test_visualize_one_correspondence.py`

**Interfaces:**
- Consumes: Task 1 adapter and existing inference/render helpers.
- Produces: CLI `--dataset {ircadb,in_silico,in_vitro}`, automatic roots/sample defaults, and non-overwriting P2P artifact names.

- [x] Add failing tests for dataset defaults, output naming, and physical GT restoration.
- [x] Implement a shared inference core that selects only one dataset item and preserves strict checkpoint loading.
- [x] Re-run focused tests.

### Task 3: Execute both real P2P visualizations and verify

**Files:**
- Create at runtime: `results/correspondence_visualization/ours_in_silico_correct_only.{png,pdf}`
- Create at runtime: `results/correspondence_visualization/in_silico_correspondence_stats.json`
- Create at runtime: `results/correspondence_visualization/ours_in_vitro_correct_only.{png,pdf}`
- Create at runtime: `results/correspondence_visualization/in_vitro_correspondence_stats.json`

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: six verified artifacts from real Ours outputs.

- [x] Run the script once for each P2P dataset with the epoch-150 cooperative checkpoint.
- [x] Verify PNG decode, PDF parse/render, JSON counts/metrics, GT-only classification, and line endpoints after render-only offset.
- [x] Visually inspect both PNGs and PDF previews.
- [x] Run focused and full project test suites; report warnings/failures exactly.
