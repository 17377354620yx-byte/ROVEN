# Best-3 FPS1024 Correspondence Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对已选中的 3 个 in-silico 与 3 个 in-vitro 样本，在原始毫米坐标、normalization 之前分别对 source/target 做确定性 FPS（最多 1024 点），重新运行同一组模型并生成隔离的对比结果。

**Architecture:** 新增独立 FPS 数据物化器，从已有 `selection.json` 读取固定样本并创建新的小型数据根目录；所有点对齐字段使用同一 FPS 索引，GT marker 与刚体信息保持不变。现有多模型 runner 与统一 correspondence collector 不改模型逻辑，只通过新的输出根目录读取 FPS 数据；新 wrapper 串联数据准备、推理、缓存和绘图。

**Tech Stack:** Python 3.10、NumPy、pytest、Bash、现有多环境模型 runner、matplotlib。

**Spec:** 用户 2026-10-01 请求：“将这几个样本用最远点下采样的方式采样到1024个点，看看各个模型的效果”。

## Global Constraints

- source 与 target 分别在原始毫米坐标中、模型 normalization 之前 FPS。
- FPS 不允许通过重复点上采样；原始点数少于 1024 时保留全部点。因此 `00049/00000/00001.npz` 的 target 保留真实 927 点。
- 初始 FPS 点固定为距点云质心最远点，保证确定性和可复现性。
- 所有原始结果不覆盖；新输出根目录名包含 `fps1024`。
- GT marker、GT 变换语义和 5 mm correct correspondence 判定不变。
- 全部 5 mm correct correspondence 继续绘制，不限制线数。
- Go-ICP 不产生 correspondence，继续明确标记 N/A。

## Review Focus

- 点数不足 1024：必须保留原点且不复制；Task 1 测试覆盖。
- 重复点与稳定 tie：相同输入必须给出相同索引且索引唯一；Task 1 测试覆盖。
- in-silico 点对齐数组：`flow` 与数字 noise 数组必须使用相应 source/target FPS 索引；Task 2 测试覆盖。
- mesh 拓扑：保留的 face/edge 索引必须重映射且不越界；Task 2 测试覆盖。
- GT marker/变换信息：物化前后逐值不变；Task 2 测试覆盖。

---

### Task 1: 确定性最远点采样核心

**Files:**
- Create: `tools/prepare_visibility_fps.py`
- Create: `tests/test_prepare_visibility_fps.py`

**Interfaces:**
- Produces: `farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray`
- Produces: `remap_faces(faces: np.ndarray, selected: np.ndarray) -> np.ndarray`
- Produces: `remap_edges(edges: np.ndarray, selected: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write failing tests** for deterministic FPS, no duplicates, farthest coverage, and `N < count` passthrough.
- [ ] **Step 2: Run** `python -m pytest tests/test_prepare_visibility_fps.py -v` and verify missing API failures.
- [ ] **Step 3: Implement** the three interfaces with finite `(N,3)` validation and deterministic centroid-farthest initialization.
- [ ] **Step 4: Run the test file** and verify PASS.

### Task 2: 物化两种 P2P FPS 数据集

**Files:**
- Modify: `tools/prepare_visibility_fps.py`
- Modify: `tests/test_prepare_visibility_fps.py`

**Interfaces:**
- Consumes: Task 1 FPS and topology remapping APIs.
- Produces: `materialize_sample(dataset, source_path, output_path, count) -> dict`
- Produces: `build_fps_selection(selection_path, output_root, count) -> dict`

- [ ] **Step 1: Write failing fixture tests** for in-silico/in-vitro aligned arrays, topology bounds, unchanged markers, lists/statistics/original indices, manifest counts, and nested sample paths.
- [ ] **Step 2: Run the targeted tests** and verify expected failures.
- [ ] **Step 3: Implement sample materialization and selection generation.**
- [ ] **Step 4: Run the test file** and verify PASS.

### Task 3: FPS1024 一键运行与隔离输出

**Files:**
- Create: `scripts/run_visibility_best3_fps1024_correspondences.sh`
- Modify: `scripts/run_visibility_best3_correspondences.sh`
- Create: `tests/test_run_visibility_best3_fps1024.py`

**Interfaces:**
- Consumes: Task 2 `selection.json` and existing `run_visibility_best3_correspondences.sh`.
- Produces: `output/visualization/visibility_0.2_0.4_correspondence_best3_fps1024/` 下的缓存、PNG、PDF、CSV 与 manifest。

- [ ] **Step 1: Write failing dry-run/command tests** for output isolation, FPS count, existing model runner reuse, and renderer invocation.
- [ ] **Step 2: Run targeted tests** and verify failure.
- [ ] **Step 3: Implement wrapper script** with `P2P_GPU`/`P2P_FORCE` passthrough and resumability；现有 runner 接受 `P2P_PREPARED_SELECTION=1`，使 `P2P_FORCE=1` 时不重建已物化的 FPS selection。
- [ ] **Step 4: Run tests** and verify PASS.

### Task 4: 真实推理、绘图与验收

**Files:**
- Modify: `docs/VISIBILITY_BEST3_CORRESPONDENCE.md`
- Outputs only: `output/visualization/visibility_0.2_0.4_correspondence_best3_fps1024/`

**Interfaces:**
- Consumes: Task 3 wrapper.
- Produces: 两张 3×10 PNG/PDF、逐格 PNG、54 个真实 correspondence cache、合并 metrics CSV。

- [ ] **Step 1: Run** `P2P_GPU=1 bash scripts/run_visibility_best3_fps1024_correspondences.sh`.
- [ ] **Step 2: Verify data counts**: source 1024；target 为 `min(original_count, 1024)`；无重复索引；GT markers unchanged.
- [ ] **Step 3: Verify artifacts and semantics**: 54 caches GT 复算一致，rendered correct count 等于 `<5 mm` 数量，60 panels，PNG/PDF 可打开。
- [ ] **Step 4: Visually inspect PNG/PDF** for fixed view, line attachment, white background, no axes, and target visibility.
- [ ] **Step 5: Run targeted and full pytest suites**, document actual metrics and output paths.
