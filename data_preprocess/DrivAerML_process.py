#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrivAerML small surf+vol preprocessing

What this script does:
- For run_id in [1..max_runs]
  - check required surface/volume part0 files exist
  - skip runs listed in --processed_skip
  - load surface points/normals/pressure (downsampled)
  - load volume cell centers/pressure/velocity (downsampled)
  - apply coordinate transform (axis permutation + shift + scale)
  - compute unsigned SDF + direction for volume points w.r.t. surface points
  - crop volume points with an axis-aligned box
  - build features:
      volume: [x,y,z,sdf,dirx,diry,dirz]
      surface:[x,y,z,0,nx,ny,nz]
    and concatenate them into x
  - build supervision:
      volume: [p, ux, uy, uz]
      surface:[p, 0, 0, 0]
    and concatenate them into y
  - keep last --keep_last_n rows (matches your original [-70000:])
  - save x_{run_id}.npy and y_{run_id}.npy into --save_root

Outputs:
  save_root/
    x_1.npy, y_1.npy, x_2.npy, y_2.npy, ...

Notes:
- Only uses "part0" for both surface and volume.
- Comments are in English per your request.
"""

import argparse
import os
from typing import Set, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------- Core math helpers ----------------
def transform(
        surf_points: np.ndarray,
        surf_normals: np.ndarray,
        vol_points: np.ndarray,
        target_len: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply your original coordinate transform:

    Positions:
      (x,y,z) -> (x,z,y)

    Normals:
      (nx,ny,nz) -> (nx,nz,ny)

    Then:
      - shift along new Y so min(Y)=0
      - scale by target_len / (extent of new Z)
      - center along new Z by subtracting mean(Z)
    """
    surf_points = np.asarray(surf_points, dtype=np.float64)
    surf_normals = np.asarray(surf_normals, dtype=np.float64)
    vol_points = np.asarray(vol_points, dtype=np.float64)

    new_surf_pos = np.empty_like(surf_points, dtype=np.float64)
    new_surf_nrm = np.empty_like(surf_normals, dtype=np.float64)
    new_vol_pos = np.empty_like(vol_points, dtype=np.float64)

    # positions: (x,y,z) -> (x,z,y)
    new_surf_pos[:, 0] = surf_points[:, 0]
    new_surf_pos[:, 1] = surf_points[:, 2]
    new_surf_pos[:, 2] = surf_points[:, 1]

    new_vol_pos[:, 0] = vol_points[:, 0]
    new_vol_pos[:, 1] = vol_points[:, 2]
    new_vol_pos[:, 2] = vol_points[:, 1]

    # normals: (nx,ny,nz) -> (nx,nz,ny)
    new_surf_nrm[:, 0] = surf_normals[:, 0]
    new_surf_nrm[:, 1] = surf_normals[:, 2]
    new_surf_nrm[:, 2] = surf_normals[:, 1]

    bound_max = np.max(new_surf_pos, axis=0)
    bound_min = np.min(new_surf_pos, axis=0)

    # shift along new Y
    new_surf_pos[:, 1] -= bound_min[1]
    new_vol_pos[:, 1] -= bound_min[1]

    # scale using extent of new X
    length = float(bound_max[0] - bound_min[0])
    if length <= 1e-12:
        raise RuntimeError(f"Degenerate extent for scaling: {length}")
    scale = float(target_len / length)

    new_surf_pos *= scale
    new_vol_pos *= scale

    # center along new Z
    z_avg = float(np.mean(new_surf_pos[:, 0]))
    new_surf_pos[:, 0] -= z_avg
    new_vol_pos[:, 0] -= z_avg

    return new_surf_pos, new_surf_nrm, new_vol_pos


def get_sdf(target: np.ndarray, boundary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute unsigned distance to nearest boundary point, and direction vector.

    Returns:
      dists: (N,)
      dirs : (N,3)
    """
    boundary = np.asarray(boundary, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(boundary)
    dists, idx = nn.kneighbors(target)
    nearest = boundary[idx[:, 0]]

    dirs = (target - nearest) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def filter_box(data: np.ndarray, sup: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points by an axis-aligned box (your original thresholds).
    """
    x_min, x_max = -3.0, 4.2
    y_min, y_max = 0.0, 2.5
    z_min, z_max = -1.5, 1.5

    mask = (
            (data[:, 0] >= x_min) & (data[:, 0] <= x_max) &
            (data[:, 1] >= y_min) & (data[:, 1] <= y_max) &
            (data[:, 2] >= z_min) & (data[:, 2] <= z_max)
    )
    return data[mask], sup[mask]


# ---------------- IO helpers ----------------
def parse_int_set(csv: str) -> Set[int]:
    """
    Parse comma-separated integers into a set.
    """
    csv = (csv or "").strip()
    if not csv:
        return set()
    return {int(x.strip()) for x in csv.split(",") if x.strip()}


def required_paths(surf_root: str, vol_root: str, run_id: int) -> Tuple[str, str]:
    """
    Return the two existence-check paths used by your original script.
    """
    vol_center_path = os.path.join(vol_root, f"run_{run_id}", f"run_{run_id}_cell_centers_part0.npy")
    surf_points_path = os.path.join(surf_root, f"run_{run_id}", f"boundary_{run_id}_points_part0.npy")
    return vol_center_path, surf_points_path


def load_run_part0(
        surf_root: str,
        vol_root: str,
        run_id: int,
        surf_step: int,
        vol_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load surface + volume arrays for one run_id (part0) with downsampling.
    """
    # ---- Surface ----
    surf_points = np.load(
        os.path.join(surf_root, f"run_{run_id}", f"boundary_{run_id}_points_part0.npy")
    )[::surf_step, :]
    surf_normals = np.load(
        os.path.join(surf_root, f"run_{run_id}", f"boundary_{run_id}_normals_part0.npy")
    )[::surf_step, :]
    surf_p = np.load(
        os.path.join(surf_root, f"run_{run_id}", f"boundary_{run_id}_pMeanTrim_part0.npy")
    )[::surf_step, None]
    surf_v = np.zeros((surf_p.shape[0], 3), dtype=surf_p.dtype)
    surf_super = np.concatenate((surf_p, surf_v), axis=-1)  # (Ns,4)

    # ---- Volume ----
    vol_points = np.load(
        os.path.join(vol_root, f"run_{run_id}", f"run_{run_id}_cell_centers_part0.npy")
    )[::vol_step, :]
    vol_p = np.load(
        os.path.join(vol_root, f"run_{run_id}", f"run_{run_id}_pMeanTrim_part0.npy")
    )[::vol_step, None]
    vol_v = np.load(
        os.path.join(vol_root, f"run_{run_id}", f"run_{run_id}_UMeanTrim_part0.npy")
    )[::vol_step, :]
    vol_super = np.concatenate((vol_p, vol_v), axis=-1)  # (Nv,4)

    return surf_points, surf_normals, surf_super, vol_points, vol_super


# ---------------- Main pipeline ----------------
def run_preprocess(args) -> None:
    os.makedirs(args.save_root, exist_ok=True)

    processed_skip = parse_int_set(args.processed_skip)

    missing_data: List[int] = []
    remained_data: List[int] = []

    for run_id in range(1, args.max_runs + 1):
        vol_center_path, surf_points_path = required_paths(args.surf_root, args.vol_root, run_id)

        if (not os.path.exists(vol_center_path)) or (not os.path.exists(surf_points_path)):
            missing_data.append(run_id)
            print(f"[Missing] run_{run_id}")
            continue

        if run_id in processed_skip:
            print(f"[Skip processed] run_{run_id}")
            continue

        out_x = os.path.join(args.save_root, f"x_{run_id}.npy")
        out_y = os.path.join(args.save_root, f"y_{run_id}.npy")
        if args.skip_existing and os.path.exists(out_x) and os.path.exists(out_y):
            print(f"[Skip existing] run_{run_id}")
            continue

        remained_data.append(run_id)

        # Load arrays
        surf_points, surf_normals, surf_super, vol_points, vol_super = load_run_part0(
            args.surf_root, args.vol_root, run_id, args.surf_step, args.vol_step
        )

        # Transform
        surf_points, surf_normals, vol_points = transform(
            surf_points, surf_normals, vol_points, target_len=args.target_len
        )

        # SDF for volume points
        vol_sdf, vol_dir = get_sdf(vol_points, surf_points)

        # Volume features: [xyz, sdf, dir]
        init_ext = np.c_[vol_points, vol_sdf, vol_dir]  # (Nv,7)

        # Crop volume points
        init_ext, vol_super = filter_box(init_ext, vol_super)

        # Surface features: [xyz, 0, normal]
        init_surf = np.c_[
            surf_points,
            np.zeros((surf_points.shape[0], 1), dtype=surf_points.dtype),
            surf_normals,
        ]  # (Ns,7)

        # Concat (volume first, then surface)
        x = np.concatenate((init_ext, init_surf), axis=0)
        y = np.concatenate((vol_super, surf_super), axis=0)

        # Keep last N rows (matches your original behavior)
        if args.keep_last_n > 0:
            x = x[-args.keep_last_n:, :]
            y = y[-args.keep_last_n:, :]

        np.save(out_x, x)
        np.save(out_y, y)

        print(f"[OK] run_{run_id} saved: x={x.shape} y={y.shape}")

    print("\n========== Summary ==========")
    print("missing_data:", missing_data)
    print("remained_data:", remained_data)
    print("len(remained_data):", len(remained_data))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DrivAerML preprocess (surface+volume part0 -> x_i/y_i)"
    )

    # ---- Configurable roots (what you asked for) ----
    parser.add_argument(
        "--surf_root",
        type=str,
        default="Surface_data/",
        help="Root directory of DrivAerML surface data",
    )
    parser.add_argument(
        "--vol_root",
        type=str,
        default="Volume_data/",
        help="Root directory of DrivAerML volume data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./drivAerml_npys",
        help="Output directory to save x_{run_id}.npy and y_{run_id}.npy",
    )

    # ---- Run control ----
    parser.add_argument("--max_runs", type=int, default=500, help="Process run_1..run_{max_runs}")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output x/y already exist")

    # ---- Downsampling ----
    parser.add_argument("--surf_step", type=int, default=10, help="Surface downsampling step")
    parser.add_argument("--vol_step", type=int, default=8, help="Volume downsampling step")

    # ---- Transform / selection ----
    parser.add_argument("--target_len", type=float, default=5.0, help="Target length used in scaling")
    parser.add_argument("--keep_last_n", type=int, default=70000, help="Keep last N rows; set <=0 to disable")

    # ---- Skip list ----
    parser.add_argument(
        "--processed_skip",
        type=str,
        default="",
        help="Comma-separated run IDs that have already been processed and should be skipped",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run_preprocess(args)


if __name__ == "__main__":
    main()
