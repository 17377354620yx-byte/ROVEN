# P2I-LReg synthetic rigid registration

This experiment implements the audited `released_corrected_v1` protocol. The
source is the complete preoperative liver, the reference is the released
synthetic partial liver, and every transform maps `src -> ref`. All internal
coordinates are metres.

The released synthetic lists contain placeholder PLY files whose only vertex
is `(0, 0, 0)`. They do not contain a registrable intraoperative observation
and are deterministically excluded before the train/validation partition,
using point-cloud content only (no GT pose or GT correspondences). This leaves
45,558 training pairs, 500 validation pairs and 2,794 test pairs. Every run
manifest records the filtering policy and excluded case IDs.

All architectures use exactly the same dataset adapter, 8192-point sampling,
1 mm voxel hierarchy, predicted correspondences, top-k selection, metrics and
pose estimators. `geotransformer` disables RTOR and A3; `rtor` enables RTOR and
disables A3; `rtor_a3` is the complete RTORv2 model and enables both RTOR and
the A3 fine local refiner. Inference removes `transform` before model forward.
There is no ICP, GT-guided inference, oracle selection, SRSA or non-rigid
deformation.

The single-pair liver model uses micro-batch 1 with two-step gradient
accumulation, giving effective batch size 2. The 8192-point/1 mm hierarchy is
memory intensive, so this experiment records and shares `angle_k=1` and
transformer hidden dimension 128 across all architectures. Selective BF16 is
restricted to the geometric-transformer core; its outputs are converted back
to FP32 before RTOR, losses, correspondence processing and pose estimation.

Run the mandatory direction/unit check first:

```bash
conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/check_dataset.py \
  --num_samples 100 --output output/p2i_lreg_dataset_sanity_100.json
```

Formal training (120 epochs by default):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
  conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/trainval.py \
  --architecture geotransformer

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
  conda run -n geo_v2 python experiments/geotransformer.p2i_lreg/trainval.py \
  --architecture rtor

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
  P2I_LREG_RUN_NAME=rtor_a3 conda run -n geo_v2 python \
  experiments/geotransformer.p2i_lreg/trainval.py --architecture rtor_a3
```

The complete RTORv2 run can also be started with
`bash scripts/train_p2i_lreg_rtor_v2.sh`. Its checkpoints are written under
`output/geotransformer.p2i_lreg.rtor_a3/snapshots/`. Add `--resume` to the
script command to continue from that run's `snapshot.pth.tar`.

Evaluation defaults to all configured top-k values and all three estimators:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n geo_v2 python \
  experiments/geotransformer.p2i_lreg/test.py --architecture rtor_a3 \
  --snapshot output/geotransformer.p2i_lreg.rtor_a3/snapshots/epoch-120.pth.tar
```
