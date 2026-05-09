# LeGo-FLoc

<div align="center">

## LeGo-FLoc: Learning 3D Geometric Priors and Room Style Knowledge for Adaptive Visual Floorplan Localization

**Bolei Chen** &nbsp;&middot;&nbsp; **Shiyong Meng** &nbsp;&middot;&nbsp; **Jiaxu Kang** &nbsp;&middot;&nbsp; **Ping Zhong<sup>&dagger;</sup>** &nbsp;&middot;&nbsp; **Yixiong Liang** &nbsp;&middot;&nbsp; **Xinwang Liu** &nbsp;&middot;&nbsp; **Jianxin Wang<sup>&dagger;</sup>**

<sup>&dagger;</sup> Corresponding authors

</div>

LeGo-FLoc studies visual floorplan localization with three trainable/evaluable variants:

- `3DP`: uses a ResNet50 depth predictor initialized from 3D geometric prior weights.
- `RSK`: uses the same mono-depth checkpoint layout initialized from room style knowledge weights.
- `fusion`: freezes a same-dataset RSK expert (`mv_net`) and 3DP expert (`mono_net`), then trains a selector to adaptively fuse their predicted depth distributions.

The repository exposes one training entrypoint and three dataset-specific evaluation entrypoints for `Gibson`, `Structured3D/S3D`, and `PALMS`.

## Installation

```bash
git clone <this-repo-url> LeGo-FLoc
cd LeGo-FLoc
conda create -n lego_floc python=3.8 -y
conda activate lego_floc
pip install -r requirements.txt
```

Use the PyTorch/CUDA build that matches your machine. Experiments in this repository are expected to run on GPU.

## Data Preparation

Datasets and checkpoints are intentionally not stored in git. Put or symlink them under the repository root with the following names:

```text
LeGo-FLoc/
  datasets_s3d/
    Structured3D/
      split.yaml
      <scene_id>/
        imgs/
        poses_map.txt
        depth40.txt
    desdf/
  datasets_gibson/
    gibson_f/
      split.yaml
      <scene_id>/
        rgb/
        poses.txt
        depth40.txt
    desdf/
  datasets_palms_main/
    full_s3d_like_oracle_noflip_rgborder/
      split_68train_12test.yaml
      <session_id>/
        imgs/
        poses_map.txt
        depth40.txt
    full_s3d_like_desdf_hardclip20/
  checkpoints/
    resnet50_3DPrior.pth
    resnet50_RSK.pth
```

Recommended symlink layout on our server:

```bash
ln -s /home/ysk/meng/floDiffLoc/datasets_s3d datasets_s3d
ln -s /home/ysk/meng/floDiffLoc/datasets datasets_gibson
ln -s /path/to/datasets_palms_main datasets_palms_main
ln -s /path/to/checkpoints checkpoints
```

Download and preprocess `Structured3D/S3D` and `Gibson` following the dataset instructions in [F3Loc](https://github.com/felix-ch/f3loc). Download and prepare `PALMS` following the [PALMS repository](https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones). After preprocessing, keep the directory names above or update the corresponding YAML config paths.

Dataset-specific evaluation settings are fixed in the configs:

- `Gibson`: single-frame evaluation, `ray_v=11`, `fwidth=3/8`, map resolution `0.01 m/pixel`.
- `S3D`: single-frame evaluation, `FOV=80`, `ray_v=9`, corrected localization unit `0.1 m`.
- `PALMS`: frame and `session360` evaluation, default mode `both`.

## Checkpoints

Pretrained weights and released LeGo-FLoc checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1A9DbJgx7Ih0U5eoYb65bP7d86_svTf4R?usp=sharing).

The configs expect these initialization weights:

```text
checkpoints/resnet50_3DPrior.pth
checkpoints/resnet50_RSK.pth
```

Training saves checkpoints with the same module prefixes used by the evaluated LeGo-FLoc checkpoints:

```text
3DP / RSK: encoder.depth_feature.*
fusion: comp_d_net.mv_net.*, comp_d_net.mono_net.*, comp_d_net.selector.*
```

For `fusion`, first train or provide the same-dataset `3DP` and `RSK` mono checkpoints, then set these fields in the fusion config:

```yaml
expert_3dp_checkpoint: checkpoints/LeGo_ckpts/3dp_S3D/mono.ckpt
expert_rsk_checkpoint: checkpoints/LeGo_ckpts/RSK_S3D/mono.ckpt
```

## Configuration

The public config interface is method- and dataset-oriented:

```yaml
method: 3DP        # 3DP | RSK | fusion
dataset:
  type: s3d        # gibson | s3d | palms
  data_folder: datasets_s3d/Structured3D
```

Main experiment configs are under `configs/experiments/`:

```text
gibson_3dp.yaml      gibson_rsk.yaml      gibson_fusion.yaml
s3d_3dp.yaml         s3d_rsk.yaml         s3d_fusion.yaml
palms_3dp.yaml       palms_rsk.yaml       palms_fusion.yaml
```

## Training

Run training with:

```bash
python training/train_depth_model.py --config configs/experiments/s3d_3dp.yaml
```

S3D:

```bash
python training/train_depth_model.py --config configs/experiments/s3d_3dp.yaml
python training/train_depth_model.py --config configs/experiments/s3d_rsk.yaml
python training/train_depth_model.py --config configs/experiments/s3d_fusion.yaml
```

Gibson:

```bash
python training/train_depth_model.py --config configs/experiments/gibson_3dp.yaml
python training/train_depth_model.py --config configs/experiments/gibson_rsk.yaml
python training/train_depth_model.py --config configs/experiments/gibson_fusion.yaml
```

PALMS:

```bash
python training/train_depth_model.py --config configs/experiments/palms_3dp.yaml
python training/train_depth_model.py --config configs/experiments/palms_rsk.yaml
python training/train_depth_model.py --config configs/experiments/palms_fusion.yaml
```

Logs and checkpoints are written to:

```text
logs/runs/<run_name>/checkpoints/
```

For a quick sanity check:

```bash
python training/train_depth_model.py --config configs/experiments/s3d_3dp.yaml --fast-dev-run
```

## Evaluation

Each dataset has a separate evaluation script. The checkpoint should match the method selected in the config.

S3D single-frame evaluation:

```bash
python eval/eval_s3d.py \
  --config configs/experiments/s3d_3dp.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt

python eval/eval_s3d.py \
  --config configs/experiments/s3d_rsk.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt

python eval/eval_s3d.py \
  --config configs/experiments/s3d_fusion.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt
```

Gibson single-frame evaluation:

```bash
python eval/eval_gibson.py \
  --config configs/experiments/gibson_3dp.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt

python eval/eval_gibson.py \
  --config configs/experiments/gibson_rsk.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt

python eval/eval_gibson.py \
  --config configs/experiments/gibson_fusion.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt
```

PALMS frame + session360 evaluation:

```bash
python eval/eval_palms.py \
  --config configs/experiments/palms_3dp.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt \
  --mode both

python eval/eval_palms.py \
  --config configs/experiments/palms_rsk.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt \
  --mode both

python eval/eval_palms.py \
  --config configs/experiments/palms_fusion.yaml \
  --checkpoint logs/runs/<run_name>/checkpoints/<checkpoint>.ckpt \
  --mode both
```

`--mode frame` can be used for PALMS frame-only evaluation. `Gibson` and `S3D` are frame-only.

## Repository Layout

```text
lego_floc/                 Core datasets, models, fusion, localization evaluator
training/train_depth_model.py
                           Training entrypoint
eval/eval_gibson.py        Gibson evaluation
eval/eval_s3d.py           S3D evaluation
eval/eval_palms.py         PALMS evaluation
configs/experiments/       Paper experiment configs
```
