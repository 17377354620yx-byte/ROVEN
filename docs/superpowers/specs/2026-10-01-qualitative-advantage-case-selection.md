# P2P 定性优势样本筛选规范

## 目标

在 `in_silico_visibility_0.2_0.4` 与 `in_vitro_visibility_0.2_0.4` 中，各选择 3 个在 FPS1024 correspondence correct-only 图中，Ours 相比所有可输出 correspondence 的基线具有明确视觉优势的真实样本。

## 科研表述边界

- 输出必须标记为“定性优势样本（selected qualitative examples）”，不得将其解释为整体性能结论。
- 5 mm 内点阈值、GT 变换、原始毫米坐标与固定视角均保持不变。
- 不通过改变阈值、隐藏基线 correspondence 或限制基线线数制造优势。
- Go-ICP 仅输出刚体变换、不输出 correspondence，因此在 correspondence 图中保持 N/A，不参与“正确线数量”排序。

## 两阶段固定筛选

### 阶段 A：完整 benchmark 建立候选池

可比较方法为 Ours、GeoTransformer、CASTV2、DFAT、Lepard、PARENet、LiverMatch、LiverMatch+P2P。候选必须满足 Ours 的 `rms_tre_mm` 严格小于其余 7 种方法。

为兼顾配准质量与 correct-only 图的可辨识度，分别计算：

1. `rms_margin_mm = min(baseline_rms_tre_mm) - ours_rms_tre_mm`；
2. `ir_gap = ours_IR - max(baseline_IR)`；
3. 对带有 `fine_corr_count` 的 Ours、GeoTransformer、PARENet，使用 `IR × fine_corr_count` 作为高输出数量方法的正确线代理量，并计算 Ours 相对两者最大值的 margin。

候选池取以上三个指标各自降序前 4 名的并集，再按三项名次和补齐到每个数据集 12 个；排序及 tie-break 必须确定性。

### 阶段 B：FPS1024 真实 correspondence 验证

候选样本按现有确定性 FPS 规则将 source/target 各采样至最多 1024 点，并对 9 种可输出 correspondence 的方法真实推理。correct correspondence 仍由原始毫米坐标下 `GT distance < 5 mm` 判断，全部正确线均绘制。

最终 correct-only 视觉样本必须满足：

1. Ours 的 `num_correct_correspondences` 严格大于其余 8 种方法；
2. 所有方法结果状态完整且样本 ID 对齐；
3. FPS1024 实际 `RMSE_mm` 必须完整审计并输出，但它不是 correct-only 图的视觉入选条件。优先选择 Ours 实际 RMSE 同时严格最优的样本。

合格样本先按 Ours 的 FPS1024 实际 RMSE 是否严格最优排序，再按 Ours 对第二名的正确线数量 margin、真实 RMSE margin、Ours IR、sample ID 排序，各取前 3。若任一数据集不足 3 个，流程必须失败并如实报告，不允许降低 5 mm 正确线标准。

该调整源于 60+60 候选的真实推理证据：in-silico 存在 5 个“正确线最多且实际 RMSE 最低”的交集样本；in-vitro 的 60 个完整 benchmark 严格优胜候选中恰有 3 个“正确线最多”样本，但不存在“正确线最多且 FPS1024 实际 RMSE 最低”的交集。用户要求的是 correct-only 图的视觉优势，因此 RMSE 作为独立审计信息，不得把 in-vitro 样本描述为 FPS1024 最终配准误差第一。

## 输出

- 候选选择、FPS1024 数据、各模型原始导出、统一 correspondence cache；
- 每个数据集最终 3×10 PNG/PDF、逐格 PNG、CSV、manifest；
- `qualitative_selection_report.json`，记录候选、剔除原因、最终样本及各方法数值。
