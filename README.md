# LeGo-FLoc

LeGo-FLoc is the clean experiment repository for the 3DP, RSK, and fusion localization pipeline. It keeps one training entrypoint, three dataset-specific evaluation entrypoints, and paper-facing configs only.

## Methods

- `3DP`: ResNet50 initialized from 3DPrior weights.
- `RSK`: ResNet50-IBN initialized from RSK weights.
- `fusion`: freezes a trained 3DP expert and a trained RSK expert, then trains a selector to fuse their predicted depth distributions.

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

Fusion configs require trained expert checkpoints, for example:

```yaml
expert_3dp_checkpoint: checkpoints/experts/s3d_3dp.ckpt
expert_rsk_checkpoint: checkpoints/experts/s3d_rsk.ckpt
```

## Evaluation

```bash
python eval/eval_s3d.py --config configs/experiments/s3d_3dp.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt
python eval/eval_gibson.py --config configs/experiments/gibson_rsk.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt
python eval/eval_palms.py --config configs/experiments/palms_fusion.yaml --checkpoint logs/runs/<run>/checkpoints/<ckpt>.ckpt --mode both
```

## Repository Hygiene

This repository intentionally excludes unrelated baselines, historical experiment folders, generated logs, local datasets, and model weights. Only LeGo-FLoc training, evaluation, configs, and reusable utilities should be pushed to GitHub.
