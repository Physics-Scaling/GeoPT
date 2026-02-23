#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DTCHull end-to-end preprocess (NO intermediate saving)

- Download GeoPT/DTCHull from Hugging Face (or use local data_root)
- For each i in [i_start, i_end]:
  1) Find volume VTK (DTCHull_500_*.vtk) and hull VTK (hull_500_*.vtk) that belong to case i
  2) Load with PyVista (HF-safe: force_ext to avoid blob-without-extension issue)
  3) Extract one bounding box region for volume points
  4) Build intermediate arrays in memory
  5) Apply transform + SDF + filter and concatenate (volume + hull)
  6) Save final x_i.npy, y_i.npy, cond_i.npy (cond: degrees)

Outputs (per i):
  - x_{i}.npy : (N_total, 7)  [x,y,z, sdf(or 0 for surface), nx,ny,nz]
  - y_{i}.npy : (N_total, 4)  [p, Ux, Uy, Uz]
  - cond_{i}.npy : (1,) degrees  (converted from radians parsed from filename)

Notes:
- This script assumes you can infer cond (radians) from the last '_' token in the *volume vtk filename*.
  If your naming differs, adjust parse_cond_from_filename().
- Pattern matching tries two strategies:
  A) case index appears in a parent folder name (e.g., .../hull_12/DTCHull_500_*.vtk)
  B) case index appears in the filename itself
"""

import argparse
import os
import glob
import math
import numpy as np
import pyvista as pv
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def read_case_pair(data_root: str, i: int, dtc_glob: str, hull_glob: str):
    """
    Expected layout:
      {data_root}/hull_{i}/DTCHull_500_*.vtk
      {data_root}/hull_{i}/hull_500_*.vtk
    """
    case_dir = os.path.join(data_root, f"hull_{i}")
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case folder not found: {case_dir}")

    dtc_matches = sorted(glob.glob(os.path.join(case_dir, dtc_glob)))
    hull_matches = sorted(glob.glob(os.path.join(case_dir, hull_glob)))

    if not dtc_matches:
        raise FileNotFoundError(f"No DTC VTK found in {case_dir} with pattern {dtc_glob}")
    if not hull_matches:
        raise FileNotFoundError(f"No Hull VTK found in {case_dir} with pattern {hull_glob}")

    # Keep your style: if multiple matches, use the first
    if len(dtc_matches) > 1:
        print(f"[Info] Multiple DTC matches for i={i}, using: {dtc_matches[0]}")
    if len(hull_matches) > 1:
        print(f"[Info] Multiple Hull matches for i={i}, using: {hull_matches[0]}")

    vol_path = dtc_matches[0]
    hull_path = hull_matches[0]

    vol = pv_read_hf_safe(vol_path)  # or pv.read(vol_path) if purely local
    hull = pv_read_hf_safe(hull_path)  # same

    return vol, vol_path, hull, hull_path


# ---------------- HF download ----------------
def hf_download_subdir(
        repo_id: str,
        subdir: str,
        repo_type: str = "dataset",
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
) -> str:
    """Download only a subdir from HF repo into local cache and return snapshot root."""
    from huggingface_hub import snapshot_download

    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=[f"{subdir}/**"],
        cache_dir=cache_dir,
        revision=revision,
        resume_download=True,
    )
    return local_root


# ---------------- VTK read helpers (HF-safe) ----------------
def pv_read_hf_safe(path: str) -> pv.DataSet:
    """
    HF LFS cache may resolve to blob files without extension.
    Force reader by using the extension from the *logical* filename.
    """
    path = os.path.abspath(str(path)).strip().rstrip("\x00")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".vtk", ".vtm", ".vtu", ".vtp"):
        ds = pv.read(path, force_ext=ext)
    else:
        ds = pv.read(path)
    if isinstance(ds, pv.MultiBlock):
        ds = ds.combine()
    return ds


def read_one(pattern: str) -> Tuple[pv.DataSet, str]:
    """
    Match your style: matches = sorted(glob.glob(pattern)); use matches[0].
    recursive=True allows data_root layout changes.
    """
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"Cannot find file matching pattern: {pattern}")
    if len(matches) > 1:
        print(f"[Info] Multiple matches for {pattern}, using: {matches[0]}")
    return pv_read_hf_safe(matches[0]), matches[0]


def ensure_point_array(ds: pv.DataSet, name: str) -> pv.DataSet:
    """
    Ensure array is in point_data. If in cell_data, convert using VTK filter (more stable).
    """
    if name in ds.point_data:
        return ds
    if name in ds.cell_data:
        import vtk

        f = vtk.vtkCellDataToPointData()
        f.SetInputData(ds)
        f.PassCellDataOn()
        f.Update()
        out = pv.wrap(f.GetOutput())
        if name not in out.point_data:
            raise KeyError(
                f"Converted cell->point but '{name}' still not in point_data. "
                f"point={list(out.point_data.keys())}, cell={list(out.cell_data.keys())}"
            )
        return out
    raise KeyError(f"Field '{name}' not found. Available arrays: {ds.array_names}")


def get_normals_if_any(ds: pv.DataSet):
    for key in ["Normals", "normals", "Normal", "normal"]:
        if key in ds.point_data:
            arr = np.asarray(ds.point_data[key])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr, True
    return None, False


def compute_surface_normals(poly: pv.PolyData) -> np.ndarray:
    p2 = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    if "Normals" not in p2.point_data:
        raise RuntimeError("Failed to compute normals on hull surface.")
    return np.asarray(p2.point_data["Normals"])


def in_box(points: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return (
            (points[:, 0] >= bmin[0]) & (points[:, 0] <= bmax[0]) &
            (points[:, 1] >= bmin[1]) & (points[:, 1] <= bmax[1]) &
            (points[:, 2] >= bmin[2]) & (points[:, 2] <= bmax[2])
    )


def parse_cond_from_filename(path: str) -> float:
    """
    Your original: float(name[:-4].split("_")[-1])  # radians
    Example: DTCHull_500_xxx_0.5235987756.vtk -> parse last token as float.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    tok = base.split("_")[-1]
    return float(tok)


# ---------------- Geometry transforms & SDF ----------------
def transform(surf_points, surf_normals, vol_points):
    new_surf_pos = np.zeros(surf_points.shape)
    new_surf_normal = np.zeros(surf_normals.shape)
    new_vol_pos = np.zeros(vol_points.shape)

    new_surf_pos[:, 0] = -surf_points[:, 0]
    new_surf_pos[:, 1] = surf_points[:, 2]
    new_surf_pos[:, 2] = surf_points[:, 1]

    new_surf_normal[:, 0] = -surf_normals[:, 0]
    new_surf_normal[:, 1] = surf_normals[:, 2]
    new_surf_normal[:, 2] = surf_normals[:, 1]

    new_vol_pos[:, 0] = -vol_points[:, 0]
    new_vol_pos[:, 1] = vol_points[:, 2]
    new_vol_pos[:, 2] = vol_points[:, 1]

    bound_min = np.min(new_surf_pos, axis=0)

    # shift "z" in your code (axis=1)
    new_surf_pos[:, 1] -= bound_min[1]
    new_vol_pos[:, 1] -= bound_min[1]

    # fixed scale (waterline heuristic)
    scale = 5 / 7
    new_surf_pos[:, :3] *= scale
    new_vol_pos[:, :3] *= scale

    # shift x
    x_avg = np.mean(new_surf_pos[:, 0:1])
    new_surf_pos[:, 0:1] -= x_avg
    new_vol_pos[:, 0:1] -= x_avg

    # shift y in your code (axis=2)
    z_avg = np.mean(new_surf_pos[:, 2:3])
    new_surf_pos[:, 2:3] -= z_avg
    new_vol_pos[:, 2:3] -= z_avg

    return new_surf_pos, new_surf_normal, new_vol_pos


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def filter_box(data, sup):
    x_min, x_max = -4.0, 5.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.5, 1.5

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    mask = (
            (x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max)
    )
    return data[mask], sup[mask]


# ---------------- One-case pipeline (in-memory stage1 + stage2) ----------------
def process_one_case(
        data_root: str,
        i: int,
        dtc_glob: str,
        hull_glob: str,
        box_min: np.ndarray,
        box_max: np.ndarray,
        p_field: str,
        u_field: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: x, y, cond_deg
      x: (N,7)  [x,y,z, sdf(or 0 for hull), nx,ny,nz]
      y: (N,4)  [p, Ux, Uy, Uz]
      cond_deg: (1,) degrees
    """

    vol, vol_path, hull, hull_path = read_case_pair(
        data_root=data_root,
        i=i,
        dtc_glob=dtc_glob,
        hull_glob=hull_glob,
    )

    if not isinstance(hull, pv.PolyData):
        hull = hull.extract_surface()

    print(f"[Loaded] i={i}")
    print(f"  vol : {vol_path}")
    print(f"  hull: {hull_path}")
    print(f"  volume points={vol.n_points}, cells={vol.n_cells}")
    print(f"  hull   points={hull.n_points}, cells={hull.n_cells}")
    print(f"  box min={box_min.tolist()}  max={box_max.tolist()}")

    # required fields (volume)
    vol = ensure_point_array(vol, p_field)
    vol = ensure_point_array(vol, u_field)

    # hull: need p_field as point_data
    if p_field not in hull.array_names:
        raise KeyError(f"Hull missing '{p_field}'. Available: {hull.array_names}")
    if p_field in hull.cell_data and p_field not in hull.point_data:
        hull = ensure_point_array(hull, p_field)

    # normals
    hull_n, hull_has_n = get_normals_if_any(hull)
    if not hull_has_n:
        hull_n = compute_surface_normals(hull)
        print("[Info] Hull normals not found -> computed point normals.")
    else:
        print("[Info] Hull normals found in VTK.")

    # select volume points in one box
    pts = np.asarray(vol.points)
    mask = in_box(pts, box_min, box_max)
    idx = np.where(mask)[0]
    print(f"[Select] volume points in box: {idx.size} / {vol.n_points}")
    if idx.size == 0:
        raise RuntimeError("No volume points inside the box. Check BOX_MIN/BOX_MAX.")

    vol_pos = pts[idx]
    hull_pos = np.asarray(hull.points)
    hull_norm = np.asarray(hull_n)

    p_vol = np.asarray(vol.point_data[p_field])[idx].reshape(-1, 1)
    U_vol = np.asarray(vol.point_data[u_field])[idx]
    y_vol = np.hstack([p_vol, U_vol])  # (Nv, 4)

    p_hull = np.asarray(hull.point_data[p_field]).reshape(-1, 1)
    y_hull = np.hstack([p_hull, np.zeros((p_hull.shape[0], 3), dtype=p_hull.dtype)])  # (Ns,4)

    cond_rad = np.array([parse_cond_from_filename(vol_path)], dtype=np.float32)
    cond_deg = (cond_rad / math.pi) * 180.0

    # Stage-2 (in memory)
    surf_points, surf_normals, vol_points = transform(hull_pos, hull_norm, vol_pos)
    vol_sdf, vol_normal = get_sdf(vol_points, surf_points)

    init_ext = np.c_[vol_points, vol_sdf, vol_normal]  # (Nv,7)
    init_ext, y_vol = filter_box(init_ext, y_vol)

    init_surf = np.c_[surf_points, np.zeros((surf_points.shape[0], 1), dtype=surf_points.dtype), surf_normals]  # (Ns,7)

    x = np.concatenate((init_ext, init_surf), axis=0)
    y = np.concatenate((y_vol, y_hull), axis=0)

    return x, y, cond_deg


def main():
    parser = argparse.ArgumentParser(description="GeoPT/DTCHull preprocess (NO intermediate saving)")

    # HF
    parser.add_argument("--hf_repo", type=str, default="HaixuWu/GeoPT")
    parser.add_argument("--hf_subdir", type=str, default="DTCHull")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_revision", type=str, default=None)

    # Local override
    parser.add_argument("--data_root", type=str, default="",
                        help="If set, skip HF and use this local DTCHull root directory.")

    # Patterns
    parser.add_argument("--dtc_glob", type=str, default="DTCHull_500_*.vtk")
    parser.add_argument("--hull_glob", type=str, default="hull_500_*.vtk")

    # Fields
    parser.add_argument("--p_field", type=str, default="p_rghMean")
    parser.add_argument("--u_field", type=str, default="UMean")

    # Box
    parser.add_argument("--box_min", type=float, nargs=3, default=[-100, -100, -100])
    parser.add_argument("--box_max", type=float, nargs=3, default=[100, 100, 100])

    # Indices
    parser.add_argument("--i_start", type=int, default=1)
    parser.add_argument("--i_end", type=int, default=130)

    # Output
    parser.add_argument("--outdir", type=str, default="./DTCHull_full_size/",
                        help="Where to save final x_i/y_i/cond_i")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip i if x_i.npy/y_i.npy/cond_i.npy already exist")

    args = parser.parse_args()

    # resolve data_root
    if args.data_root:
        data_root = args.data_root
        print("[Local] data_root:", data_root)
    else:
        local_root = hf_download_subdir(
            repo_id=args.hf_repo,
            subdir=args.hf_subdir,
            repo_type="dataset",
            cache_dir=args.hf_cache_dir,
            revision=args.hf_revision,
        )
        data_root = os.path.join(local_root, args.hf_subdir)
        print("[HF] Local root:", local_root)
        print("[HF] data_root :", data_root)

    box_min = np.array(args.box_min, dtype=np.float32)
    box_max = np.array(args.box_max, dtype=np.float32)

    os.makedirs(args.outdir, exist_ok=True)

    for i in range(args.i_start, args.i_end + 1):
        print(f"\n============[Info] Processing {i}============")

        x_path = os.path.join(args.outdir, f"x_{i}.npy")
        y_path = os.path.join(args.outdir, f"y_{i}.npy")
        c_path = os.path.join(args.outdir, f"cond_{i}.npy")

        if args.skip_existing and os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(c_path):
            print(f"[Skip] exists: {x_path}")
            continue

        x, y, cond_deg = process_one_case(
            data_root=data_root,
            i=i,
            dtc_glob=args.dtc_glob,
            hull_glob=args.hull_glob,
            box_min=box_min,
            box_max=box_max,
            p_field=args.p_field,
            u_field=args.u_field,
        )

        np.save(x_path, x)
        np.save(y_path, y)
        np.save(c_path, cond_deg)

        print(f"[OK] Saved final i={i}  x={x.shape} y={y.shape} cond(deg)={cond_deg[0]}")


if __name__ == "__main__":
    main()
