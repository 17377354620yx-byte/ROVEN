# Qualitative Advantage Case Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在两个 visibility 0.2–0.4 数据集中，各找出 3 个经 FPS1024 真实推理验证、Ours 的 5 mm 正确线数量严格优于所有可比 correspondence 基线的定性样本，并完整审计 RMSE 后生成固定视角对比图。

**Architecture:** 新增一个确定性的两阶段选择器：先从完整 benchmark 构建 12 个候选，再从统一 correspondence summary 中执行严格验收并输出最终 3 个样本。新增 wrapper 复用现有 FPS1024、模型 runner、collector 和 matplotlib renderer；只将现有 Predator runner 的固定 3 样本限制改为由候选子集自然控制。

**Tech Stack:** Python 3.10、NumPy、pytest、Bash、现有多环境模型推理、matplotlib。

**Spec:** `docs/superpowers/specs/2026-10-01-qualitative-advantage-case-selection.md`

## Global Constraints

- 当前 dirty checkout 原地执行，只修改本计划列出的文件，保留其他改动。
- 所有候选与最终图使用确定性 FPS1024；不足 1024 点不重复采样。
- correct correspondence 固定为原始毫米坐标中 GT residual `< 5 mm`。
- 不限制正确线数量；不修改固定相机、颜色、offset 与遮挡处理。
- Go-ICP 保持 N/A，不虚构 correspondence。
- 最终图属于挑选后的定性例证，不作为整体性能证明。

## Review Focus

- 不同方法 JSON 字段名差异（`sample`/`sample_name`、`rms_tre_mm`/`RMS_TRE_mm`）必须正确归一化；Task 1 测试覆盖。
- 缺失、重复、失败或非有限 benchmark 行必须被拒绝；Task 1 测试覆盖。
- 候选排名与 tie-break 必须确定性；Task 1 测试覆盖。
- Predator 必须运行全部候选，而不是静默截断为 3；Task 2 dry-run/文本测试覆盖。
- 最终不足 3 个严格合格样本时必须失败，不能自动放宽；Task 3 测试覆盖。

---

### Task 1: Benchmark 候选选择器

**Files:**
- Create: `tools/select_visibility_qualitative_cases.py`
- Create: `tests/test_select_visibility_qualitative_cases.py`

**Interfaces:**
- Produces: `normalize_summary(summary: Mapping) -> dict[str, dict]`
- Produces: `select_benchmark_candidates(summaries: Mapping[str, Mapping], count: int = 12, per_metric: int = 4) -> list[dict]`
- Reuses: `tools.prepare_visibility_best3.build_subset`

- [ ] **Step 1:** 先写字段归一化、严格 RMS winner、三指标候选并集、确定性补齐及无足够候选报错测试。
- [ ] **Step 2:** 运行 `python -m pytest tests/test_select_visibility_qualitative_cases.py -v`，确认因接口缺失而失败。
- [ ] **Step 3:** 实现最小候选选择与 CLI，写出候选 `selection.json` 和两个真实子集。
- [ ] **Step 4:** 重跑目标测试并确认通过。

### Task 2: 任意候选数的模型 runner 与一键候选推理

**Files:**
- Modify: `scripts/run_visibility_best3_correspondences.sh`
- Create: `scripts/run_visibility_qualitative_candidates.sh`
- Modify: `tests/test_run_visibility_best3_correspondences.py`
- Create: `tests/test_run_visibility_qualitative_candidates.py`

**Interfaces:**
- Consumes: Task 1 candidate selection、现有 `prepare_visibility_fps.py`、统一 runner。
- Produces: `output/visualization/visibility_0.2_0.4_qualitative_candidates_fps1024/` 下全部真实缓存。

- [ ] **Step 1:** 先写测试，要求 Predator 命令不含固定 `--test-limit 3`，并验证 wrapper 的隔离根目录、FPS1024、prepared selection 和后续严格选择/渲染命令。
- [ ] **Step 2:** 运行两个目标测试文件，确认预期失败。
- [ ] **Step 3:** 移除 Predator 固定 3 条限制并实现可恢复的一键 wrapper。
- [ ] **Step 4:** 重跑目标测试并确认通过。

### Task 3: 真实 correspondence 严格验收与最终选择

**Files:**
- Modify: `tools/select_visibility_qualitative_cases.py`
- Modify: `tests/test_select_visibility_qualitative_cases.py`

**Interfaces:**
- Produces: `select_verified_cases(candidate_cases, method_summaries, top_k=3) -> tuple[list[dict], list[dict]]`
- Produces: CLI `--finalize-root` 写出 `final_selection.json` 与 `qualitative_selection_report.json`。

- [ ] **Step 1:** 先写全部方法完整、Ours 5 mm 正确线严格最多、RMSE 独立审计与优先排序及不足 3 条报错测试。
- [ ] **Step 2:** 运行目标测试，确认新接口缺失失败。
- [ ] **Step 3:** 实现真实 summary 验收、剔除原因审计和最终 selection/report 写出。
- [ ] **Step 4:** 重跑目标测试并确认通过。

### Task 4: 真实运行、绘图与证据验收

**Files:**
- Modify: `docs/VISIBILITY_BEST3_CORRESPONDENCE.md`
- Outputs only: `output/visualization/visibility_0.2_0.4_qualitative_candidates_fps1024/`

**Interfaces:**
- Consumes: Tasks 1–3。
- Produces: 每个数据集 3 个最终样本、PNG/PDF/CSV/manifest/report 与逐格图。

- [ ] **Step 1:** 运行一键 wrapper，完成候选 FPS1024 与 9 个 correspondence 模型的真实推理。
- [ ] **Step 2:** 若严格条件不足 3 个，扩大阶段 A 候选数并重新推理；不得放宽阶段 B 条件。
- [ ] **Step 3:** 对最终 cache 逐项重算 `<5 mm` mask、正确线数、IR 与 RMSE 排名。
- [ ] **Step 4:** 检查 PNG/PDF 可打开、固定视角、线端附着、蓝点无遮挡、无坐标轴/网格/越界。
- [ ] **Step 5:** 运行目标测试与全量 pytest，记录真实样本、指标与绝对输出路径。
