# P2P Liver registration error visualization

This module produces the three-panel figure used for qualitative registration
analysis: ground-truth alignment, predicted alignment, and a per-source-point
error heatmap. Rendering is performed by Open3D with one shared camera; Matplotlib
composes the panels, colorbar, annotations, and distribution plots.

The default `auto` renderer uses Open3D when its installed ABI is compatible.
The supplied `geo_v2` currently combines Open3D 0.16 with NumPy 2.x, for which
the Filament renderer terminates the process; this known combination is detected
before renderer creation and uses a deterministic perspective fallback. Upgrade
Open3D (or use NumPy 1.x) and pass `--renderer open3d` to require Open3D rendering.

## Data flow

```text
source (N, 3), target (M, 3), T_pred, T_gt
                    |
                    v
pred = source @ R_pred.T + t_pred
gt   = source @ R_gt.T   + t_gt
                    |
                    v
error_i = ||pred_i - gt_i|| * scale_factor   [mm]
                    |
                    v
Matplotlib colormap -> Open3D point colors -> shared-camera renders
                    |
                    v
*_heatmap.png + *_error.npy + *_error_distribution.png
```

`scale_factor=1` means the coordinates are already in millimetres. For the
project's normalized P2P data it must be `physical_scale`, i.e. the complete
source liver radius computed by `dataset.py`.

## Coordinate and metric conventions

Both project transforms map **source to target**. The codebase represents points
as row vectors, so applying a transform is `points @ R.T + t`.

The heatmap follows the requested transform-discrepancy definition on source
surface points:

`||T_pred(source_i) - T_gt(source_i)||`.

This is not numerically identical to the evaluation RMS-TRE for the released
in-silico data. That data contains non-rigid deformation, and `test.py` evaluates
paired volumetric fiducials as

`sqrt(mean(||T_pred(source_marker_i) - target_marker_i||^2)) * physical_scale`.

`T_gt` is the least-squares rigid transform of those marker pairs. The native
batch tool recomputes the evaluation expression and requires it to match the
JSON value (default tolerance: `1e-3` mm), while keeping the surface heatmap
statistics clearly separate.

## Single case API

```python
from visualization import generate_error_heatmap

result = generate_error_heatmap(
    source, target, T_pred, T_gt,
    scale_factor=physical_scale,
    output_path="output/visualization/error_heatmap/case_0001_heatmap.png",
)
```

The returned dictionary contains output paths, the plotted color range, and
min/max/mean/median/95th-percentile/RMS statistics.

## Generic JSON batch input

Inline matrices or paths to `.npy`, `.npz`, `.txt`, `.csv`, `.json`, `.ply`, and
`.pcd` are supported. For an `.npz` with multiple arrays, add the corresponding
`src_key`, `tgt_key`, `T_pred_key`, or `T_gt_key`.

```json
{
  "cases": [{
    "case_id": "case_0001",
    "src_path": "source.npy",
    "tgt_path": "target.npy",
    "T_pred": "T_pred.npy",
    "T_gt": "T_gt.npy",
    "scale_factor": 112.4
  }]
}
```

```bash
conda run --no-capture-output -n geo_v2 \
  python tools/visualize_p2p_results.py cases.json
```

## Native project summary: top successes and failures

The existing test summary ranks successful cases (`RMS-TRE < 20 mm`) from lowest
error and failed cases from highest error. Prediction matrices must have first
been saved by `test.py --save_predictions`.

```bash
conda run --no-capture-output -n geo_v2 \
  python tools/visualize_p2p_results.py \
  output/.../in_silico_noise_none.json \
  --predictions-dir output/.../registration \
  --top-k 10
```

The batch directory also contains `error_distribution.png`, comparing aggregate
point-error densities for the selected success and failure groups, plus a JSON
manifest recording every image and the RMS-TRE consistency check.

For cross-method figures, pass one fixed `--error-range MIN MAX`; otherwise each
case uses its own observed minimum and maximum, as in an error-range-annotated
qualitative figure.
