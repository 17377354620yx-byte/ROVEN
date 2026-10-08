# Three-stage P2P liver registration visualization

`paper_registration_visualizer.py` generates one compact publication figure per
case: `(a) Input`, `(b) Registration Result`, and `(c) Error Heatmap`.

- Input: complete source in blue and partial target in red.
- Registration: `T_pred(source)` in blue and the unchanged target in red.
- Heatmap: only the extracted partial source region, colored with a monotonic
  `medical_blue` palette. Low errors are pale blue and high errors are dark,
  saturated blue; no rainbow hues are used.

## Audited project conventions

The current experiment uses `src_points` as complete/source, `ref_points` as
partial/target, and both `estimated_transform` and `transform` map complete
source to partial target. Points are transformed as `points @ R.T + t`.

Source and target are independently centered, then divided by the complete
source radius. `physical_scale` is that radius in millimetres and restores all
visualization distances to mm.

The project evaluation RMS-TRE is computed on paired volumetric markers. The
heatmap instead uses the requested directed surface distance:

```text
d_i = min_j ||predicted_partial_i - GT_partial_j|| * physical_scale
```

Native batch mode independently recomputes evaluation RMS-TRE and checks it
against the summary JSON before drawing. Predictions re-run on GPU can differ
slightly from an older summary; the default accepted drift is `0.05 mm` and can
be changed with `--metric-tolerance`.

## Partial-region extraction

The source is complete, so its visible partial region is reconstructed by:

1. mapping source with `T_gt` into target coordinates;
2. estimating target surface spacing;
3. retaining source samples near the observed target surface;
4. applying `T_pred` and `T_gt` to the same retained source subset;
5. querying each predicted point's nearest neighbour on the full GT partial.

The exact source indices are saved as `case_xxxxx_partial_indices.npy`.

## Python API

```python
from visualization import generate_paper_registration_visualization

generate_paper_registration_visualization(
    source_complete, target_partial, T_pred, T_gt,
    scale_factor=physical_scale,
    case_id="case_0001",
    colormap="medical_blue",
    dpi=600,
)
```

Input and Registration share the exact camera and zoom. Heatmap uses the same
camera direction but fits tightly around the partial region to avoid empty space.

## Native batch mode

The summary JSON does not contain matrices, so first save predictions with
`test.py --save_predictions`, then run:

```bash
conda run --no-capture-output -n geo_v2 \
  python visualization/paper_registration_visualizer.py \
  output/.../in_silico_noise_none.json \
  --predictions-dir output/.../registration \
  --best 5 --worst 5 \
  --output-dir output/visualization/paper_registration
```

`best` selects the lowest evaluation RMS-TRE cases; `failure` selects the
highest. Outputs are written below `best/` and `failure/`.

## Generic manifest

```json
{
  "cases": [{
    "case_id": "case_0001",
    "src_path": "source.npy",
    "tgt_path": "target.npy",
    "T_pred": "T_pred.npy",
    "T_gt": "T_gt.npy",
    "scale_factor": 98.7
  }]
}
```

Paths are resolved relative to the manifest. Each case produces:

- `case_xxxxx_heatmap.png`;
- `case_xxxxx_error.npy`;
- `case_xxxxx_partial_indices.npy`;
- `case_xxxxx_statistics.json` containing min/max, mean, RMS and P95.

Open3D is the preferred off-screen renderer; Matplotlib composes the panels and
colorbar. Because the current environment combines Open3D 0.16 with NumPy 2.x,
`--renderer auto` selects the safe fallback. Use `--renderer open3d` after
installing a compatible Open3D/NumPy combination.

The default `medical_blue` palette is defined locally for reproducible paper
figures. Standard Matplotlib sequential maps such as `Blues`, `viridis`, or
`magma` remain selectable with `--colormap`.
