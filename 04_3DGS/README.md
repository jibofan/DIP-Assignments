# Assignment 4 — Simplified 3D Gaussian Splatting

---

## Requirements

To install the required packages:

```bash
conda create -n dip python=3.10 -y
conda activate dip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python tqdm matplotlib
conda install -c conda-forge colmap
```

For Task 3, run:

```bash
python -m pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

---

## Running

Recover the camera intrinsic and extrinsic parameters, and obtain a set of sparse 3D points for initializing 3DGS:

```bash
python mvs_with_colmap.py --data_dir data/chair
```

Reproject the recovered 3D points back to each camera view to check whether the camera calibration is correct:

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

The output files will be written to `data/chair/sparse/0/`, which will be used by the subsequent training stage.

To train the customized 3DGS model, run:

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

After training is complete, generate the rendered video:

```bash
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000199.pt \
    --num_frames 240 --fps 30
```

To train the official 3DGS model, run:

```bash
python official/train.py \
    --colmap_dir data/chair \
    --checkpoint_dir data/chair/official \
    --save_every 1 --num_epochs 200
```

## Results
The execution results for customized and official 3dgs are generated and saved in the [customized](./data/chair/checkpoints) and [official](./data/chair/official) directory.

During training the model and analyzing the result, I discovered that official 3dgs is much faster than customized 3dgs, while rendering quality are nearly the same.