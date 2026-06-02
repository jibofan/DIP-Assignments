"""Render a horizontal-orbit multi-view video from a trained 3DGS checkpoint.

The camera path is a uniform horizontal circle around the scene center:
  * scene center = centroid of COLMAP 3D points
  * up axis = direction of least variance of training camera centers (SVD)
  * orbit radius / elevation = average of training cameras projected onto
    horizontal plane / up axis
  * each frame uses an OpenCV/COLMAP look-at toward the scene center

Usage:
    python render_3dgs_mv.py --colmap_dir data/chair \\
        --checkpoint data/chair/checkpoints/checkpoint_000060.pt \\
        --output data/chair/render_mv.mp4
"""

import argparse
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def look_at_colmap(eye, target, up):
    """OpenCV/COLMAP convention (x right, y down, z forward). Returns R(3,3), t(3,)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)              # camera forward
    x = np.cross(z, up)
    x /= np.linalg.norm(x)              # camera right
    y = np.cross(z, x)                  # camera down (z × x in right-handed)

    R = np.stack([x, y, z], axis=0)     # rows: camera axes expressed in world
    t = -R @ eye
    return R.astype(np.float32), t.astype(np.float32)


def build_horizontal_orbit(dataset, num_frames):
    """Generate (R, t) for cameras moving uniformly along a horizontal circle.

    Up axis is recovered from the training cameras' own y-axes (each camera's
    R[1, :] is the world direction of "image down"). NeRF-synthetic renders are
    captured upright, so this average aligns with world gravity and is far more
    reliable than the SVD-of-positions heuristic.
    """
    Rs = np.stack([np.asarray(d['R']) for d in dataset.camera_data], axis=0)
    ts = np.stack([np.asarray(d['t']).reshape(3) for d in dataset.camera_data], 0)
    cam_centers = np.einsum('nij,nj->ni', Rs.transpose(0, 2, 1), -ts)  # (N, 3)

    # Scene center: centroid of reconstructed 3D points
    scene_center = dataset.points3D_xyz.numpy().mean(axis=0)

    # Up axis: negate the average of each camera's y-axis (image down) in world
    cam_down = Rs[:, 1, :]                                       # (N, 3)
    up = -cam_down.mean(axis=0)
    up /= np.linalg.norm(up)

    # Horizontal basis perpendicular to up (Gram-Schmidt from an arbitrary seed)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, up)) > 0.95:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    # Orbit elevation + radius (averages over training cameras)
    centered = cam_centers - scene_center
    heights = centered @ up                                      # (N,)
    elevation = float(heights.mean())
    horiz_vecs = centered - np.outer(heights, up)
    radius = float(np.linalg.norm(horiz_vecs, axis=1).mean())

    orbit_center = scene_center + elevation * up

    Rs_out, ts_out = [], []
    for i in range(num_frames):
        theta = 2.0 * np.pi * i / num_frames
        eye = orbit_center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        R, t = look_at_colmap(eye, scene_center, up)
        Rs_out.append(R)
        ts_out.append(t)
    return np.stack(Rs_out), np.stack(ts_out)


def parse_args():
    p = argparse.ArgumentParser(description='Render a horizontal-orbit video from a trained 3DGS model')
    p.add_argument('--colmap_dir', type=str, required=True,
                   help='COLMAP data dir (with sparse/0_text/ and images/)')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to a trained 3DGS checkpoint (.pt)')
    p.add_argument('--output', type=str, default=None,
                   help='Output mp4 path (default: <colmap_dir>/render_mv.mp4)')
    p.add_argument('--num_frames', type=int, default=240,
                   help='Total number of frames in the video')
    p.add_argument('--fps', type=int, default=30, help='Output frame rate')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = ColmapDataset(args.colmap_dir)
    sample = dataset[0]
    H, W = sample['image'].shape[:2]
    K = sample['K'].to(device)

    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    renderer = GaussianRenderer(H, W).to(device)
    with torch.no_grad():
        gp = model()

    print(f'Building horizontal orbit from {len(dataset)} training cameras → {args.num_frames} frames')
    R_path, t_path = build_horizontal_orbit(dataset, args.num_frames)

    output = args.output or os.path.join(args.colmap_dir, 'render_mv.mp4')
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W, H))
    for i in tqdm(range(args.num_frames), desc='Rendering'):
        R = torch.as_tensor(R_path[i], device=device)
        t = torch.as_tensor(t_path[i], device=device)
        with torch.no_grad():
            frame = renderer(
                means3D=gp['positions'], covs3d=gp['covariance'],
                colors=gp['colors'], opacities=gp['opacities'],
                K=K, R=R, t=t,
            )
        frame = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    print(f'Video saved to: {output}')


if __name__ == '__main__':
    main()
