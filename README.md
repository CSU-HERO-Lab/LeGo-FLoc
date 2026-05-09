# LeGo-FLoc

LeGo-FLoc is the clean experiment repository for the 3DP, RSK, and fusion localization pipeline. It keeps one training entrypoint, three dataset-specific evaluation entrypoints, and paper-facing configs only.

## Methods

- `3DP`: ResNet50 initialized from 3DPrior weights.
- `RSK`: legacy mono-depth ResNet50 key layout initialized from compatible RSK weights.
- `fusion`: freezes a trained RSK expert (`mv_net`) and a trained 3DP expert (`mono_net`), then trains the legacy `selector` to fuse their predicted depth distributions.

## Data And Checkpoints

Local paths are ignored by git:

```bash
ln -s /home/ysk/meng/floDiffLoc/datasets_s3d datasets_s3d
ln -s /home/ysk/meng/floDiffLoc/datasets datasets_gibson
ln -s /path/to/datasets_palms_main datasets_palms_main
ln -s /path/to/checkpoints checkpoints
```

S3D uses `FOV=80` with `ray_v=9`. Gibson uses single-frame `ray_v=11` and `fwidth=3/8`. PALMS supports frame and `session360` evaluation.

## Training

```bash
python training/train_depth_model.py --config configs/experiments/s3d_3dp.yaml
python training/train_depth_model.py --config configs/experiments/s3d_rsk.yaml
python training/train_depth_model.py --config configs/experiments/s3d_fusion.yaml
python training/train_depth_model.py --config configs/experiments/gibson_3dp.yaml
python training/train_depth_model.py --config configs/experiments/gibson_rsk.yaml
python training/train_depth_model.py --config configs/experiments/gibson_fusion.yaml
python training/train_depth_model.py --config configs/experiments/palms_3dp.yaml
python training/train_depth_model.py --config configs/experiments/palms_rsk.yaml
python training/train_depth_model.py --config configs/experiments/palms_fusion.yaml
```

Fusion configs require trained expert checkpoints. The S3D/Gibson example configs point at the current local LeGo checkpoints; for new runs, replace these paths with the matching same-dataset 3DP and RSK `mono.ckpt` files.

LeGo-FLoc saves checkpoints with the same legacy module prefixes as the evaluated checkpoints:

```text
3DP / RSK: encoder.depth_feature.*
fusion: comp_d_net.mv_net.*, comp_d_net.mono_net.*, comp_d_net.selector.*
```

## Evaluation

```bash
python eval/eval_s3d.py --config configs/experiments/s3d_3dp.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt
python eval/eval_gibson.py --config configs/experiments/gibson_rsk.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt
python eval/eval_palms.py --config configs/experiments/palms_fusion.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt --mode both
```

## Repository Hygiene

This repository intentionally excludes unrelated baselines, historical experiment folders, generated logs, local datasets, and model weights. Only LeGo-FLoc training, evaluation, configs, and reusable utilities should be pushed to GitHub.
