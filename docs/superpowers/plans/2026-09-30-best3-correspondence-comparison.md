# P2P Top-3 Correspondence Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在两个 visibility 0.2–0.4 数据集上固定 Ours 最优 3 个样本，运行 9 个能产生匹配的学习方法及 Go-ICP，生成两张 3×10 的真实 correct-correspondence 论文对比图。

**Architecture:** 复用已验证的 Top-3 选择与数据子集；为每个原生推理入口增加可选的 correspondence 导出，不改变默认评测行为；由 RTORv2 中央收集器统一恢复毫米坐标、用 GT 的 5 mm 阈值计算 correct mask；最后由独立 matplotlib 渲染器画出全部正确线并生成审计文件。Go-ICP 保留为 N/A 列，不合成匹配。

**Tech Stack:** Python 3.10、PyTorch、NumPy、Matplotlib、pytest、Bash、各方法现有 Conda 环境。

**Spec:** `docs/superpowers/specs/2026-09-30-best3-correspondence-comparison-design.md`

## Global Constraints

- 固定使用设计文档中的 6 个样本，不重新针对 baseline 选样。
- correct correspondence 仅使用原始 mm 坐标、GT source-to-target 变换和严格 `< 5.0 mm` 判定。
- 所有正确线全部绘制，不设置数量上限；offset 只用于显示。
- 绝不通过 GT、预测变换或后验空间近邻为任何方法构造预测 correspondence。
- 不覆盖旧的 `visibility_0.2_0.4_livermatch_style` 结果，不修改无关工作区文件。
- 每个外部仓库改动均为默认关闭的导出参数，不能影响其原始测试结果。

---

### Task 1: 统一 correspondence 缓存、坐标恢复和 GT 判定

**Files:**
- Create: `tools/correspondence_cache.py`
- Create: `tests/test_correspondence_cache.py`

**Interfaces:**
- `restore_points_to_mm(points, metadata) -> np.ndarray`
- `compute_gt_inlier_mask(src_corr_mm, tgt_corr_mm, gt_transform_mm, threshold_mm=5.0) -> np.ndarray`
- `write_correspondence_cache(path, payload, provenance) -> None`
- `read_correspondence_cache(path) -> dict`

- [ ] **Step 1: 编写失败测试**，覆盖米/毫米恢复、中心化/缩放逆变换、source-to-target GT、严格 5 mm 边界、NaN/形状错误、offset 不影响 mask。
- [ ] **Step 2: 运行测试确认失败。**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_correspondence_cache.py -v`

Expected: FAIL，模块尚不存在。

- [ ] **Step 3: 实现最小缓存模块**，统一保存原始 mm 点云、预测点对、可选置信度、GT/预测变换、correct mask 和 provenance；读取时再次校验统计一致性。
- [ ] **Step 4: 运行 Task 1 测试。**

Expected: PASS。

### Task 2: 为九个学习方法导出真实预测 correspondence

**Files:**
- Modify: `experiments/geotransformer.p2p_liver/test.py`
- Modify: `/home/yangx/code/new_deform/GeoTransformer_base/experiments/geotransformer.p2p_liver/test_standard.py`
- Modify: `/home/yangx/code/Luo/DFAT-main/experiments/geotransformer.p2p_liver/test.py`
- Modify: `/home/yangx/code/point_clould/PARENet/experiments/P2P/test_standard.py`
- Modify: `/home/yangx/code/Luo/CASTv2/test_p2p_liver.py`
- Modify: `/home/yangx/code/new_deform/Lepard/evaluate_p2p.py`
- Modify: `/home/yangx/code/OverlapPredator/test_p2p_liver.py`
- Modify: `/home/yangx/code/OverlapPredator/lib/p2p_liver_registration.py`
- Modify: `tools/evaluate_livermatch_visibility.py`
- Create: `tests/test_correspondence_export_contracts.py`

- [ ] **Step 1: 为各适配器编写契约测试**，验证必须包含 source/target 点云、预测 source/target 点、可选分数、样本 id、GT 与单位元数据；Go-ICP 显式声明 `supports_correspondences=false`。
- [ ] **Step 2: 运行测试确认失败。**
- [ ] **Step 3: 增加默认关闭的导出选项**，直接导出：RTOR/Geo/DFAT/PARE 的 `src_corr_points/ref_corr_points/corr_scores`，CAST 的 `corres/corr_confidence`，Lepard 的 `Matching.get_match`，Predator 特征匹配阶段的实际点对，LiverMatch 两种协议的 `details["matches"]`。
- [ ] **Step 4: 对每种方法用一个选中样本做 smoke test**，确认导出非空、点对维度对应且 checkpoint 元数据正确；检查正常测试模式指标不变。
- [ ] **Step 5: 运行契约测试。**

Expected: PASS，九个学习方法均能提供真实预测点对；Go-ICP 为明确 N/A。

### Task 3: 六样本十方法可恢复运行与集中收集

**Files:**
- Create: `tools/collect_visibility_correspondences.py`
- Create: `scripts/run_visibility_best3_correspondences.sh`
- Create: `tests/test_collect_visibility_correspondences.py`
- Create: `tests/test_run_visibility_best3_correspondences.py`

- [ ] **Step 1: 编写收集器失败测试**，覆盖九种输入格式、样本顺序、checkpoint 校验、单位恢复、5 mm 统计和 Go-ICP N/A；禁止缺失点对时退化为空间近邻。
- [ ] **Step 2: 编写 dry-run 失败测试**，断言 2 个数据集×10 方法的环境、checkpoint、样本清单和输出路径正确，Predator 存在且旧的 Lepard+P2P 不在方法列表。
- [ ] **Step 3: 实现中央收集器与运行脚本**，支持 `P2P_GPU`、`P2P_DRY_RUN`、`P2P_FORCE`，元数据一致的缓存可恢复复用；真实失败保留状态和原因。
- [ ] **Step 4: 运行单元测试与 dry-run。**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_collect_visibility_correspondences.py tests/test_run_visibility_best3_correspondences.py -v`

Run: `P2P_DRY_RUN=1 scripts/run_visibility_best3_correspondences.sh`

Expected: PASS；命令覆盖 Ours、GeoTransformer、CASTV2、DFAT、Lepard、Predator、PARENet、LiverMatch、LiverMatch+P2P、Go-ICP。

### Task 4: 3×10 matplotlib correspondence 对比渲染

**Files:**
- Create: `visualization/correspondence_method_comparison.py`
- Create: `tests/test_correspondence_method_comparison.py`

**Interfaces:**
- `render_panel(ax, cache, view, style) -> dict`
- `render_dataset_comparison(dataset, selection, cache_root, output_root) -> dict`

- [ ] **Step 1: 编写布局与线段测试**，断言列顺序、3×10 布局、红/蓝/绿颜色、全部 correct mask 线段被使用、无数量截断、线端点使用显示 offset 而 mask 使用原始坐标。
- [ ] **Step 2: 编写可读性测试**，断言绿色线 collection 的绘制层级低于蓝色 target，target 有白色 halo，且固定相机参数与现有满意单图一致。
- [ ] **Step 3: 实现 matplotlib 3D 渲染器**，生成单格 PNG、两张大图 PNG/PDF、`correspondence_metrics.csv` 与 `manifest.json`；N/A 和 FAILED 使用不同明确状态。
- [ ] **Step 4: 运行渲染测试。**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_correspondence_method_comparison.py -v`

Expected: PASS。

### Task 5: 真实运行、图像 QA 与最终交付

**Files:**
- Generated: `output/visualization/visibility_0.2_0.4_correspondence_best3/**`
- Create: `docs/VISIBILITY_BEST3_CORRESPONDENCE.md`

- [ ] **Step 1: 运行完整流水线。**

Run: `P2P_GPU=1 scripts/run_visibility_best3_correspondences.sh`

- [ ] **Step 2: 核查所有缓存**：样本一致、checkpoint 一致、九个方法来自真实模型输出、correct mask 使用 GT、offset 未参与判定；统计预测数/正确数/IR/RRE/RTE/RMS-TRE。
- [ ] **Step 3: 渲染两张大图**，所有正确线全部绘制；Go-ICP 为 N/A；失败项为 FAILED。
- [ ] **Step 4: 自动检查 PNG/PDF**：可解析、尺寸一致、无空白/越界、每格线段数等于 `num_correct_correspondences`、蓝色点绘制层级正确。
- [ ] **Step 5: 人工查看两张 PNG**，检查相机、遮挡、点云形状、标题和行列对齐；如仅需视觉参数调整，不重新改变匹配或统计。
- [ ] **Step 6: 运行完整回归测试并记录日志。**

Run: `/home/yangx/miniconda3/envs/geo_v2/bin/python -m pytest tests/test_correspondence_cache.py tests/test_correspondence_export_contracts.py tests/test_collect_visibility_correspondences.py tests/test_run_visibility_best3_correspondences.py tests/test_correspondence_method_comparison.py -v`

- [ ] **Step 7: 完成运行文档**，记录最终样本、所有 checkpoint/环境、实际命令、成功/失败状态、指标表和所有绝对输出路径。

## Execution Recommendation

建议采用当前会话原生串行执行。各模型环境和外部仓库适配存在共享接口与 GPU 顺序依赖，按 Task 1–5 逐项验收最稳妥；推理阶段可恢复，单方法失败不会丢失其余缓存。

