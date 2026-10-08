#!/usr/bin/env python3
"""Build the TABLE I experiment workbook from existing evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path("/home/yangx/code")
PDF_PATH = ROOT / "new_deform/ai_worker/Resolving_the_Ambiguity_of_Complete-to-Partial_Point_Cloud_Registration_for_Image-Guided_Liver_Surgery_With_Patches-to-Partial_Matching——2026.pdf"

BINS = [
    "[0.2, 0.3)",
    "[0.3, 0.4)",
    "[0.4, 0.5)",
    "[0.5, 0.6)",
    "[0.6, 0.7)",
    "[0.7, 0.8)",
    "[0.8, 0.9)",
    "[0.9, 1.0]",
]

METHODS = [
    {
        "name": "Ours",
        "env": "geo_v2",
        "requested": ROOT / "new_deform/ai_worker/RTORv8/output/geotransformer.p2p_liver.rtor_a3_cooperative_abl1_no_proposal_seed7351",
        "result": ROOT / "new_deform/ai_worker/RTORv8/output/geotransformer.p2p_liver.rtor_a3_cooperative_abl1_no_proposal_seed7351/evaluation_epoch150/in_silico_noise_none.json",
        "aggregate_key": "method",
    },
    {
        "name": "GeoTransformer",
        "env": "geotransformer",
        "requested": ROOT / "new_deform/GeoTransformer_base/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar",
        "result": ROOT / "new_deform/GeoTransformer_base/output/geotransformer.p2p_liver/evaluation_epoch150_standard/in_silico_noise_none.json",
        "aggregate_key": "method",
    },
    {
        "name": "CASTV2",
        "env": "overpredator_py310",
        "requested": ROOT / "Luo/CASTv2/ckpt/p2p_liver/castv2-p2p-liver-epoch-150.pth.tar",
        "result": ROOT / "Luo/CASTv2/output/p2p_liver/castv2_epoch150_in_silico_all_metrics.json",
        "aggregate_key": "CASTv2",
    },
    {
        "name": "DFAT",
        "env": "dfat_both",
        "requested": ROOT / "Luo/DFAT-main/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar",
        "result": ROOT / "Luo/DFAT-main/output/geotransformer.p2p_liver/test_in_silico_epoch150_tableVI.json",
        "aggregate_key": "GeoTransformer",
    },
    {
        "name": "Lepard",
        "env": "livermatch5090",
        "requested": ROOT / "new_deform/Lepard/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth",
        "result": ROOT / "new_deform/ai_worker/RTORv2/results/visibility_0.2_0.4_benchmark_metrics_v2/lepard/in_silico_none.json",
    },
    {
        "name": "Lepard+P2P",
        "env": "livermatch5090",
        "requested": ROOT / "new_deform/Lepard/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth",
        "result": ROOT / "new_deform/ai_worker/RTORv2/results/visibility_0.2_0.4_benchmark_metrics_v2/lepard_p2p/in_silico_none.json",
    },
    {
        "name": "PARENet",
        "env": "pare5090",
        "requested": ROOT / "point_clould/PARENet/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar",
        "result": ROOT / "point_clould/PARENet/output/P2P/evaluation_epoch150_rtorv2_standard/in_silico_noise_none.json",
        "aggregate_key": "method",
    },
    {
        "name": "LiverMatch",
        "env": "livermatch5090",
        "requested": ROOT / "point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth",
        "result": ROOT / "new_deform/ai_worker/RTORv2/results/visibility_0.2_0.4_benchmark_metrics_v2/livermatch/in_silico_none.json",
    },
    {
        "name": "LiverMatch+P2P",
        "env": "livermatch5090",
        "requested": ROOT / "point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth",
        "result": ROOT / "new_deform/ai_worker/RTORv2/results/visibility_0.2_0.4_benchmark_metrics_v2/livermatch_p2p/in_silico_none.json",
    },
]

PAPER_TABLE = {
    "Procrustes": [(3.19, 1.93), (3.44, 2.16), (3.41, 2.13), (3.42, 2.19), (3.38, 2.17), (3.41, 2.17), (3.39, 2.01), (3.31, 2.08)],
    "GO-ICP [19]": [(66.96, 17.20), (66.98, 19.52), (56.40, 23.52), (55.61, 25.41), (44.83, 29.27), (33.56, 29.94), (19.90, 24.21), (5.42, 2.65)],
    "RoITr [28]": [(20.21, 25.12), (10.77, 12.71), (7.24, 8.13), (5.60, 6.05), (4.61, 3.57), (4.19, 2.61), (3.91, 2.36), (3.75, 2.46)],
    "RegTr [26]": [(84.27, 29.46), (69.89, 30.16), (59.66, 30.45), (48.08, 30.11), (33.53, 24.36), (20.24, 11.89), (12.05, 4.16), (5.82, 2.41)],
    "LiverMatch [22]": [(12.85, 13.87), (8.09, 6.88), (6.65, 5.15), (5.65, 3.98), (4.79, 2.82), (4.24, 2.45), (3.84, 2.09), (3.56, 2.07)],
    "LiverMatch [22] + proposed P2P": [(8.74, 10.23), (6.55, 5.54), (5.50, 3.69), (5.00, 3.30), (4.40, 2.71), (4.08, 2.46), (3.78, 2.15), (3.58, 2.16)],
    "Lepard [13]": [(9.47, 14.74), (6.17, 4.83), (5.22, 3.70), (4.76, 4.33), (4.20, 2.58), (3.96, 2.43), (3.74, 2.17), (3.57, 2.22)],
    "Lepard [13] + proposed P2P": [(6.73, 5.96), (5.63, 3.72), (4.96, 3.28), (4.60, 2.93), (4.22, 2.68), (4.00, 2.50), (3.78, 2.22), (3.59, 2.25)],
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bins_from_samples(data: dict) -> list[dict | None]:
    rows = []
    for index in range(8):
        low = round(0.2 + 0.1 * index, 1)
        high = round(low + 0.1, 1)
        values = []
        for sample in data.get("samples", []):
            visibility = sample.get("visibility")
            rms_tre = sample.get("rms_tre_mm", sample.get("RMS_TRE_mm"))
            if visibility is None or rms_tre is None:
                continue
            in_bin = visibility >= low and (visibility < high if index < 7 else visibility <= high)
            if in_bin:
                values.append(float(rms_tre))
        if values:
            rows.append(
                {
                    "range": BINS[index],
                    "count": len(values),
                    "mean_rms_tre_mm": statistics.fmean(values),
                    "std_rms_tre_mm": statistics.pstdev(values),
                }
            )
        else:
            rows.append(None)
    return rows


def extract_method(method: dict) -> dict:
    data = load_json(method["result"])
    if method.get("aggregate_key"):
        aggregate = data[method["aggregate_key"]]
        raw_bins = aggregate["visibility_bins"]
        by_range = {item["range"].replace("]", ")") if item["range"] != "[0.9, 1.0]" else item["range"]: item for item in raw_bins}
        bins = [by_range.get(label) for label in BINS]
    else:
        bins = bins_from_samples(data)
        # The recent Lepard/LiverMatch benchmark was intentionally filtered to
        # visibility <= 0.4. A single boundary sample at exactly 0.4 does not
        # constitute coverage of the full [0.4, 0.5) TABLE I bin.
        if data.get("evaluated_samples", 0) < 6000:
            bins[2:] = [None] * 6
    available = sum(item is not None for item in bins)
    internal_checkpoint = data.get("checkpoint", "")
    requested = str(method["requested"])
    mismatch = bool(internal_checkpoint and internal_checkpoint != requested and method["name"] == "Ours")
    return {
        **method,
        "data": data,
        "bins": bins,
        "available": available,
        "evaluated_samples": data.get("evaluated_samples", len(data.get("samples", []))),
        "internal_checkpoint": internal_checkpoint,
        "checkpoint_mismatch": mismatch,
    }


def style_header(ws, row: int, start: int, end: int) -> None:
    fill = PatternFill("solid", fgColor="1F4E78")
    for col in range(start, end + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def style_grid(ws, min_row: int, max_row: int, min_col: int, max_col: int) -> None:
    side = Side(style="thin", color="B7B7B7")
    border = Border(left=side, right=side, top=side, bottom=side)
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True)


def build_workbook(output: Path) -> None:
    extracted = [extract_method(method) for method in METHODS]
    wb = Workbook()
    ws = wb.active
    ws.title = "TABLE1_当前结果"
    ws.merge_cells("A1:J1")
    ws["A1"] = "TABLE I — 当前模型 RMS-TRE（mm，mean ± std）"
    ws["A1"].font = Font(size=15, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="17365D")
    ws["A1"].alignment = Alignment(horizontal="center")
    ws.merge_cells("A2:J2")
    ws["A2"] = "in silico phantom；noise=None；黄色单元格表示未找到该 checkpoint 的完整评测输出"
    ws["A2"].alignment = Alignment(horizontal="center")
    ws["A2"].font = Font(italic=True, color="666666")
    headers = ["模型"] + BINS + ["覆盖情况"]
    for col, value in enumerate(headers, 1):
        ws.cell(4, col, value)
    style_header(ws, 4, 1, len(headers))
    missing_fill = PatternFill("solid", fgColor="FFF2CC")
    complete_fill = PatternFill("solid", fgColor="E2F0D9")
    for row_index, method in enumerate(extracted, 5):
        ws.cell(row_index, 1, method["name"])
        for bin_index, item in enumerate(method["bins"], 2):
            cell = ws.cell(row_index, bin_index)
            if item is None:
                cell.value = "未找到完整输出"
                cell.fill = missing_fill
            else:
                cell.value = f"{item['mean_rms_tre_mm']:.2f} ± {item['std_rms_tre_mm']:.2f}"
        coverage = "完整（8/8）" if method["available"] == 8 else f"部分（{method['available']}/8）"
        ws.cell(row_index, 10, coverage)
        ws.cell(row_index, 10).fill = complete_fill if method["available"] == 8 else missing_fill
    style_grid(ws, 4, 4 + len(extracted), 1, 10)
    ws.freeze_panes = "B5"
    ws.auto_filter.ref = f"A4:J{4 + len(extracted)}"
    ws.column_dimensions["A"].width = 22
    for col in range(2, 10):
        ws.column_dimensions[get_column_letter(col)].width = 19
    ws.column_dimensions["J"].width = 15
    ws.row_dimensions[1].height = 26

    detail = wb.create_sheet("数值明细")
    detail_headers = ["模型", "Visibility ratio", "Mean RMS-TRE (mm)", "Std RMS-TRE (mm)", "N", "覆盖状态", "结果文件"]
    detail.append(detail_headers)
    style_header(detail, 1, 1, len(detail_headers))
    for method in extracted:
        for label, item in zip(BINS, method["bins"]):
            detail.append(
                [
                    method["name"],
                    label,
                    None if item is None else item["mean_rms_tre_mm"],
                    None if item is None else item["std_rms_tre_mm"],
                    None if item is None else item["count"],
                    "已汇总" if item is not None else "缺失",
                    str(method["result"]),
                ]
            )
    style_grid(detail, 1, detail.max_row, 1, len(detail_headers))
    detail.freeze_panes = "A2"
    detail.auto_filter.ref = f"A1:G{detail.max_row}"
    for row in range(2, detail.max_row + 1):
        detail.cell(row, 3).number_format = "0.0000"
        detail.cell(row, 4).number_format = "0.0000"
        if detail.cell(row, 6).value == "缺失":
            for col in range(1, 8):
                detail.cell(row, col).fill = missing_fill
    widths = [20, 18, 22, 22, 10, 12, 90]
    for col, width in enumerate(widths, 1):
        detail.column_dimensions[get_column_letter(col)].width = width

    provenance = wb.create_sheet("数据来源")
    provenance_headers = ["模型", "Conda 环境", "用户指定模型路径", "实际结果文件", "结果文件内 checkpoint", "样本数", "覆盖", "核对说明"]
    provenance.append(provenance_headers)
    style_header(provenance, 1, 1, len(provenance_headers))
    for method in extracted:
        notes = []
        if method["available"] < 8:
            notes.append("仅找到 visibility∈[0.2,0.4] 的近期结果；高可见度区间未用 PDF 旧值填充")
        if method["checkpoint_mismatch"]:
            notes.append("结果文件位于 RTORv8 目录，但 JSON 内 checkpoint 字段仍指向 RTORv5 同名实验，请复核复制/迁移历史")
        if method["name"] == "DFAT":
            notes.append("结果 JSON 内聚合键名为 GeoTransformer；按用户指定 checkpoint 与文件路径标记为 DFAT")
        provenance.append(
            [
                method["name"],
                method["env"],
                str(method["requested"]),
                str(method["result"]),
                method["internal_checkpoint"],
                method["evaluated_samples"],
                f"{method['available']}/8 bins",
                "；".join(notes) if notes else "完整输出，checkpoint 路径已匹配",
            ]
        )
    style_grid(provenance, 1, provenance.max_row, 1, len(provenance_headers))
    provenance.freeze_panes = "A2"
    for col, width in enumerate([20, 20, 82, 98, 82, 12, 14, 62], 1):
        provenance.column_dimensions[get_column_letter(col)].width = width
    for row in range(2, provenance.max_row + 1):
        if provenance.cell(row, 7).value != "8/8 bins":
            for col in range(1, 9):
                provenance.cell(row, col).fill = missing_fill

    paper = wb.create_sheet("PDF原TABLE1")
    paper.merge_cells("A1:I1")
    paper["A1"] = "PDF TABLE I 原始数值 — RMS-TRE（mm，mean ± std）"
    paper["A1"].font = Font(size=14, bold=True, color="FFFFFF")
    paper["A1"].fill = PatternFill("solid", fgColor="7030A0")
    paper["A1"].alignment = Alignment(horizontal="center")
    paper.merge_cells("A2:I2")
    paper["A2"] = f"来源：{PDF_PATH}；印刷页 466（PDF 第 8 页）；仅作原论文对照，不与当前模型结果混填"
    paper["A2"].alignment = Alignment(horizontal="center", wrap_text=True)
    for col, value in enumerate(["Method"] + BINS, 1):
        paper.cell(4, col, value)
    style_header(paper, 4, 1, 9)
    for method_name, values in PAPER_TABLE.items():
        paper.append([method_name] + [f"{mean:.2f} ± {std:.2f}" for mean, std in values])
    style_grid(paper, 4, paper.max_row, 1, 9)
    paper.freeze_panes = "B5"
    paper.column_dimensions["A"].width = 38
    for col in range(2, 10):
        paper.column_dimensions[get_column_letter(col)].width = 18

    readme = wb.create_sheet("说明")
    notes = [
        ["项目", "内容"],
        ["目标", "按所给 PDF 的 TABLE I 结构汇总 9 个指定模型的现有结果。"],
        ["指标", "RMS-TRE on paired volumetric fiducials，单位 mm；表中为总体分布的均值 ± 总体标准差（ddof=0）。"],
        ["数据集", "in silico phantom，noise=None。"],
        ["区间", "[0.2,0.3)、[0.3,0.4)、…、[0.9,1.0]；最后一个区间包含右端点。"],
        ["完整性", "Ours、GeoTransformer、CASTV2、DFAT、PARENet：6315 个样本、8 个区间。Lepard、Lepard+P2P、LiverMatch、LiverMatch+P2P：当前仅找到 1538 个低可见度样本，因此只填前两个区间。"],
        ["缺失处理", "不使用 PDF 原论文数值填补当前 checkpoint 的缺失结果；缺失处明确标记并标黄。"],
        ["统计读取", "完整输出优先读取 JSON 已保存的 visibility_bins；低可见度输出按 samples 中 visibility 与 RMS_TRE_mm/rms_tre_mm 重算。"],
        ["PDF 结构检查", "结构预检因当前环境缺少 pypdf 返回 UNAVAILABLE；已用 pdfinfo、pdftotext 与 PDF 第 8 页图像人工交叉核对表头和原始数值。"],
        ["生成日期", "2026-09-24（Asia/Shanghai）"],
    ]
    for row in notes:
        readme.append(row)
    style_header(readme, 1, 1, 2)
    style_grid(readme, 1, readme.max_row, 1, 2)
    readme.column_dimensions["A"].width = 22
    readme.column_dimensions["B"].width = 120
    for row in range(1, readme.max_row + 1):
        readme.row_dimensions[row].height = 36

    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)

    # Re-open once to catch malformed XLSX output and critical sheet omissions.
    checked = load_workbook(output, read_only=True, data_only=False)
    expected = {"TABLE1_当前结果", "数值明细", "数据来源", "PDF原TABLE1", "说明"}
    if set(checked.sheetnames) != expected:
        raise RuntimeError(f"Unexpected workbook sheets: {checked.sheetnames}")
    if checked["TABLE1_当前结果"].max_row != 13:
        raise RuntimeError("Current-results sheet does not contain all 9 methods")
    checked.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_tables/TABLE1_in_silico_visibility_RMS_TRE_20260924.xlsx"),
    )
    args = parser.parse_args()
    build_workbook(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
