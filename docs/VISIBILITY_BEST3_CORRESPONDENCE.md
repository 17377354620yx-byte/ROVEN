# Visibility 0.2–0.4：Best-3 correspondence 对比图

## 评估与绘图协议

- 两套数据分别从 Ours 的统一基准结果中按 RMS-TRE 从低到高选择 3 个样本。
- 主判定阈值固定为 5 mm：对每条模型预测 correspondence，使用中央数据集提供的 source-to-target GT 变换，在原始毫米坐标中计算残差，`distance < 5.0` 才是 correct correspondence。
- 2/3/4/5 mm 的 IR 同时写入统计表，所有方法使用完全相同的阈值；不通过调小阈值来单独优化 Ours。
- visualization-only offset 只在完成 correct/incorrect 判定后施加，不参与残差计算。
- 每格绘制全部 5 mm correct correspondences，不按置信度或数量截断。source 为红色、target 为蓝色、correct correspondence 为绿色；target 使用白色 halo，避免绿色线覆盖蓝点。
- 固定视角为 `rx=22, ry=-18, rz=-8, elev=10, azim=-78, layout_right=1.25, layout_up=0.75`。
- Go-ICP 只输出刚体变换，不产生 predicted correspondences，因此图中明确标记为 N/A，不用最近邻伪造连线。

## 选中样本

### in-silico

| sample id | visibility | Ours selection RMS-TRE (mm) |
|---|---:|---:|
| `00008/00002/00000.npz` | 0.397333 | 0.525337 |
| `00049/00000/00001.npz` | 0.206321 | 0.563705 |
| `00046/00002/00003.npz` | 0.386601 | 0.574366 |

### in-vitro

| sample id | visibility | Ours selection RMS-TRE (mm) |
|---|---:|---:|
| `5_00086.npz` | 0.362451 | 2.009561 |
| `5_00085.npz` | 0.294200 | 2.148071 |
| `5_00087.npz` | 0.314000 | 2.327428 |

## 复现命令

模型推理与统一 correspondence 缓存：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
P2P_GPU=1 bash scripts/run_visibility_best3_correspondences.sh
```

用缓存重新绘图（不重复模型推理）：

```bash
/home/yangx/miniconda3/envs/geo_v2/bin/python \
  visualization/correspondence_method_comparison.py \
  --selection output/visualization/visibility_0.2_0.4_correspondence_best3/selection.json \
  --output-root output/visualization/visibility_0.2_0.4_correspondence_best3 \
  --dataset all \
  --dpi 300
```

## 主要输出

输出根目录：

`/home/yangx/code/new_deform/ai_worker/RTORv2/output/visualization/visibility_0.2_0.4_correspondence_best3`

- `in_silico_best3_correspondence_comparison.png/.pdf`
- `in_vitro_best3_correspondence_comparison.png/.pdf`
- `correspondence_metrics.csv`：两个数据集、9 个 correspondence 方法、6 个样本的 54 行合并统计
- `in_silico_correspondence_metrics.csv`、`in_vitro_correspondence_metrics.csv`
- `manifest.json`：视角、阈值、方法顺序、文件路径与每格渲染线数
- `panels/<dataset>/<sample>/<method>.png`：60 张逐格图（包含 Go-ICP N/A 格）
- `correspondences/`：54 个可复算的模型 correspondence 缓存及方法摘要

## Ours 的 5 mm correspondence 统计

| dataset | sample id | predicted | correct | IR | RRE (deg) | RTE (mm) | RMSE (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| in-silico | `00008/00002/00000.npz` | 3278 | 2025 | 0.617755 | 0.587206 | 0.265398 | 0.457814 |
| in-silico | `00049/00000/00001.npz` | 2804 | 1160 | 0.413695 | 0.483809 | 0.346204 | 0.494684 |
| in-silico | `00046/00002/00003.npz` | 3400 | 1498 | 0.440588 | 0.351503 | 0.318166 | 0.459626 |
| in-vitro | `5_00086.npz` | 3227 | 1527 | 0.473195 | 1.295393 | 2.805802 | 1.124023 |
| in-vitro | `5_00085.npz` | 3007 | 1392 | 0.462920 | 2.491081 | 3.972617 | 1.409515 |
| in-vitro | `5_00087.npz` | 3103 | 1442 | 0.464712 | 3.118755 | 5.442036 | 1.735378 |

注意：selection JSON 中的 `ours_rms_tre_mm` 来自用于选样的统一基准运行；上表 `RMSE_mm` 来自本次 correspondence 推理缓存中的估计变换，因此两者是不同运行阶段的审计字段，不应混为同一列。

## FPS1024 实验

运行命令：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
P2P_GPU=1 bash scripts/run_visibility_best3_fps1024_correspondences.sh
```

FPS 使用原始毫米坐标，分别作用于 source/target，初始点固定为距质心最远点。所有原始点数超过 1024 的点云均得到 1024 个无重复点；`00049/00000/00001.npz` 的 target 原本只有 927 点，因此保留全部 927 点，不通过复制点伪造 1024 点。GT markers、样本列表、统计元数据和 original indices 均保持一致。

隔离输出目录：

`/home/yangx/code/new_deform/ai_worker/RTORv2/output/visualization/visibility_0.2_0.4_correspondence_best3_fps1024`

Ours 的 FPS1024 结果：

| dataset | sample id | predicted | correct @ 5 mm | IR | RRE (deg) | RTE (mm) | RMSE (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| in-silico | `00008/00002/00000.npz` | 1453 | 293 | 0.201652 | 2.799 | 1.783 | 2.364 |
| in-silico | `00049/00000/00001.npz` | 881 | 15 | 0.017026 | 30.184 | 37.787 | 46.279 |
| in-silico | `00046/00002/00003.npz` | 1289 | 241 | 0.186967 | 4.977 | 3.281 | 5.806 |
| in-vitro | `5_00086.npz` | 1802 | 334 | 0.185350 | 4.856 | 10.652 | 3.649 |
| in-vitro | `5_00085.npz` | 1072 | 308 | 0.287313 | 3.469 | 4.995 | 2.920 |
| in-vitro | `5_00087.npz` | 1281 | 376 | 0.293521 | 1.344 | 2.510 | 1.223 |

三样本均值显示，FPS1024 下 Ours 的 IR/RMSE 为 in-silico `0.1352 / 18.150 mm`、in-vitro `0.2554 / 2.597 mm`。该输入密度与模型原训练/测试密度不同，结果不支持“下采样后 Ours 更优”的结论；尤其 in-silico 的 `00049/00000/00001.npz` 明显失效。图和 CSV 保留真实输出，没有通过阈值或选线调整掩盖退化。

## FPS1024 定性视觉优势样本

该实验先在完整 benchmark 中限定 Ours 的 RMS-TRE 严格优于 7 个已有完整结果的学习方法，再对每个数据集 60 个候选执行 9 个 correspondence 方法的真实 FPS1024 推理。最终 correct-only 图以“Ours 的 5 mm 正确线数量严格多于其余 8 个方法”为视觉入选标准；实际 FPS1024 RMSE 完整写入报告并优先用于排序，但不与正确线数量混为一个指标。

in-silico 的 3 个样本同时满足正确线最多和实际 RMSE 最低：

| sample id | visibility | Ours correct / predicted | correct-count margin | Ours RMSE / next-best RMSE (mm) |
|---|---:|---:|---:|---:|
| `00060/00001/00001.npz` | 0.218347 | 297 / 2014 | +130 | 3.314 / 3.811 |
| `00026/00004/00001.npz` | 0.218125 | 162 / 1301 | +77 | 4.674 / 5.312 |
| `00085/00001/00000.npz` | 0.253668 | 136 / 1110 | +55 | 3.661 / 6.399 |

in-vitro 的 60 个候选中仅以下 3 个样本满足正确线数量全方法第一；其 FPS1024 实际 RMSE 不是全方法第一，因此只能描述为 correspondence correct-only 视觉优势样本：

| sample id | visibility | Ours correct / predicted | correct-count margin | Ours RMSE / best baseline RMSE (mm) |
|---|---:|---:|---:|---:|
| `2_00016.npz` | 0.370413 | 280 / 1424 | +132 | 14.229 / 12.330 |
| `5_00008.npz` | 0.317494 | 299 / 1349 | +62 | 3.664 / 3.576 |
| `2_00072.npz` | 0.382266 | 247 / 1361 | +44 | 16.741 / 10.038 |

复现命令：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
P2P_GPU=1 bash scripts/run_visibility_qualitative_candidates.sh
```

隔离输出目录：

`/home/yangx/code/new_deform/ai_worker/RTORv2/output/visualization/visibility_0.2_0.4_qualitative_candidates60_fps1024`

`qualitative_selection_report.json` 保存全部 60+60 候选的入选/剔除依据及 9 个方法逐样本数值；`final_selection.json` 保存最终 3+3 样本。PNG/PDF、逐格图、CSV、manifest 与 1080 个候选 cache 均保留在同一目录。Go-ICP 无 predicted correspondence，继续标记为 N/A。
