# P2P Top-3 Correspondence Comparison Design

## 目标

在以下两个低重叠 P2P 数据集上，分别按 Ours 的真实测试 RMS-TRE 从低到高选择最优 3 个样本，并生成论文级 correct-correspondence 方法对比图：

- `/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4/`
- `/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4/`

对比列固定为：Ours、GeoTransformer、CASTV2、DFAT、Lepard、Predator、PARENet、LiverMatch、LiverMatch+P2P、Go-ICP。每个数据集输出一张 3×10 大图；每行是同一个样本，所有方法使用完全相同的 source、target、GT 和可视化参数。

## 固定选样

选样复用已完成的全量基准结果，按 Ours `rms_tre_mm` 升序排名，并固定如下：

| 数据集 | 排名 | sample id | visibility | Ours RMS-TRE (mm) |
|---|---:|---|---:|---:|
| in-silico | 1 | `00008/00002/00000.npz` | 0.397333 | 0.525337 |
| in-silico | 2 | `00049/00000/00001.npz` | 0.206321 | 0.563705 |
| in-silico | 3 | `00046/00002/00003.npz` | 0.386601 | 0.574366 |
| in-vitro | 1 | `5_00086.npz` | 0.362451 | 2.009561 |
| in-vitro | 2 | `5_00085.npz` | 0.294200 | 2.148071 |
| in-vitro | 3 | `5_00087.npz` | 0.314000 | 2.327428 |

所有方法必须处理以上固定的 6 个样本，不能按方法替换为更有利的样本。`selection.json` 保留全量结果文件、排序字段、原始索引和选样值，确保可追溯。

## 正确匹配定义

模型输出的每对预测 correspondence 记为原始毫米坐标中的 `(p_src, p_tgt)`。正确匹配只由 GT source-to-target 刚体变换判定：

```text
p_src_gt = R_gt @ p_src + t_gt
d = ||p_src_gt - p_tgt||_2
correct = d < 5.0 mm
```

IR 定义为 `num_correct / num_predicted`。SRE 是基于表面点的配准误差，是另一项配准指标，不能用于逐条 correspondence 的正确/错误判定。预测变换、RANSAC/P2P 解算结果和 visualization-only offset 均不得参与 correct mask 的计算。

各模型输出在进入统一缓存前恢复到原始毫米坐标。统一收集器必须对点数、形状、有限值、GT 方向和单位做检查，并通过人工构造点对的单元测试验证 5 mm 边界。

## 模型输出与真实性

每种学习方法必须导出其真实模型预测匹配，不允许用 GT 或几何最近邻替代模型输出：

- Ours、GeoTransformer、DFAT、PARENet：使用 `src_corr_points`、`ref_corr_points`、`corr_scores`；
- CASTV2：使用 `output["corres"]` 与 `output["corr_confidence"]`；
- Lepard：使用 `Matching.get_match(...)` 返回的预测匹配；
- Predator：保留其特征最近邻匹配阶段实际生成的 source/target 匹配及模型 overlap/saliency 分数；
- LiverMatch 与 LiverMatch+P2P：使用各自推理详情中的 `matches`，分别运行各自的配准协议；
- Go-ICP：该方法只估计刚体变换，不预测 correspondence。因此该列保留，但显示 source/target 点云和 `N/A — no predicted correspondences`，不生成绿色线，不用后验最近邻伪造匹配。

每条缓存保存 checkpoint、Conda 环境、命令、样本标识、原始点云、预测点对、置信度（若存在）、GT 变换、预测变换（若存在）以及单位恢复元数据。

## 绘图协议

每格采用 matplotlib 3D，固定以下语义和视觉设置：

- source 点云为红色；target 点云为蓝色；correct predicted correspondences 为绿色线；
- 只显示 correct correspondence；所有满足 `< 5 mm` 的正确线全部绘制，不设置 `num_rendered_correspondences` 上限，不按置信度截断；
- target 可加 visualization-only offset，使两片点云分开；offset 仅在 correct mask 计算完成后用于画图；
- 使用已经确认满意的姿态：`vis-rx=22`、`vis-ry=-18`、`vis-rz=-8`、`view-elev=10`、`view-azim=-78`、`layout-right=1.25`、`layout-up=0.75`；
- 绿色线先绘制，蓝色 target 与红色 source 后绘制；target 使用不透明点和白色 halo/描边，避免绿色线盖住蓝色点云；
- 默认线宽 `0.45`、透明度 `0.38`；所有方法使用一致参数，禁止通过改变透明度或线宽弱化某一方法；
- 白底、无坐标轴、无网格、正交投影、同行共享视角和显示范围；标题只保留方法名，左侧标注 sample id 与 overlap/visibility。

由于用户明确要求画出所有达标线，线密度将由该方法的预测 correspondence 数量和 5 mm 内点数自然决定。这是方法输出差异的一部分，不做跨方法数量均衡。为避免极密集线段遮挡点云，只通过绘制层级、透明度和 target halo 改善可读性。

## 指标与输出

对每个样本/方法记录：

- `num_source_points`
- `num_target_points`
- `num_pred_correspondences`
- `num_correct_correspondences`
- `inlier_ratio`
- `inlier_threshold_mm = 5.0`
- `RRE_deg`
- `RTE_mm`
- `RMSE_mm`（沿用现有基准的 RMS-TRE 定义并在 manifest 中明确命名）
- 状态、失败原因、checkpoint、环境、原始命令和缓存路径

新结果写入独立目录，不覆盖现有点云叠加图：

```text
output/visualization/visibility_0.2_0.4_correspondence_best3/
├── selection.json
├── correspondences/<dataset>/<sample>/<method>.npz
├── panels/<dataset>/<sample>/<method>.png
├── in_silico_best3_correspondence_comparison.png
├── in_silico_best3_correspondence_comparison.pdf
├── in_vitro_best3_correspondence_comparison.png
├── in_vitro_best3_correspondence_comparison.pdf
├── correspondence_metrics.csv
└── manifest.json
```

## 验收条件

1. 六个样本与固定选样清单完全一致；每个学习方法均来自指定 checkpoint 和环境。
2. 每个缓存中的点对能追溯到对应模型的实际输出变量；没有以 GT、预测变换或后验几何近邻伪造 correspondence。
3. correct mask 在原始毫米坐标、visualization offset 之前，严格以 GT 和 `< 5.0 mm` 计算。
4. 所有正确线均绘制；统计中的正确数与图中 Line3DCollection 的线段数一致。
5. PNG/PDF 可打开、标题不裁切、无坐标轴/网格、点云不越界，绿色线端点与红蓝点位置一致。
6. 蓝色 target 位于绿色线之上并有 halo，密集区域仍能辨认蓝色轮廓。
7. Go-ICP 明确显示 N/A；任何真实推理失败均显示 FAILED 并记录原因。

