#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import random
import numpy as np
import pyvista as pv
from typing import Optional


def read_hf_safe(path: str) -> pv.DataSet:
    path = os.path.abspath(str(path)).strip().rstrip("\x00")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".vtk", ".vtm", ".vtu", ".vtp"):
        return pv.read(path, force_ext=ext)
    return pv.read(path)


# --------- HF download ----------
def hf_download_subdir(repo_id: str, subdir: str, repo_type: str = "dataset",
                       cache_dir: Optional[str] = None, revision: Optional[str] = None) -> str:
    """
    Download only a subdir from a Hugging Face repo into local cache and return local root path.
    """
    from huggingface_hub import snapshot_download

    allow = [f"{subdir}/**"]
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow,
        cache_dir=cache_dir,
        revision=revision,
        resume_download=True,
    )
    return local_root  # contains subdir inside


# --------- safe cell->point ----------
def ensure_point_scalar(mesh: pv.DataSet, name: str) -> pv.DataSet:
    """
    Ensure scalar array `name` exists in point_data.
    If it only exists in cell_data, convert cell->point using VTK filter (more stable).
    """
    if name in mesh.point_data:
        return mesh

    if name in mesh.cell_data:
        import vtk  # local import helps in multiprocess setups
        f = vtk.vtkCellDataToPointData()
        f.SetInputData(mesh)
        f.PassCellDataOff()
        f.Update()
        mesh2 = pv.wrap(f.GetOutput())

        if name not in mesh2.point_data:
            raise KeyError(
                f"Converted cell->point, but still cannot find '{name}' in point_data. "
                f"Available point_data: {list(mesh2.point_data.keys())}, "
                f"Available cell_data: {list(mesh2.cell_data.keys())}"
            )
        return mesh2

    raise KeyError(
        f"Cannot find scalar '{name}' in either point_data or cell_data.\n"
        f"Available point_data: {list(mesh.point_data.keys())}\n"
        f"Available cell_data:  {list(mesh.cell_data.keys())}"
    )


def build_sim_map(json_path: str) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    if "simulations" not in data:
        raise KeyError(f"JSON missing key 'simulations': {json_path}")
    sim_map = {sim["folder_name"]: sim for sim in data["simulations"]}
    return sim_map


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_points_normals(points: np.ndarray, normals: np.ndarray, target_len: float = 5.0):
    new_points = np.zeros_like(points, dtype=np.float64)
    new_normals = np.zeros_like(normals, dtype=np.float64)

    new_points[:, 0] = -points[:, 0]
    new_points[:, 1] = points[:, 2]
    new_points[:, 2] = points[:, 1]

    new_normals[:, 0] = -normals[:, 0]
    new_normals[:, 1] = normals[:, 2]
    new_normals[:, 2] = normals[:, 1]

    bound_max = np.max(new_points, axis=0)
    bound_min = np.min(new_points, axis=0)

    # shift: make z start at 0  (your code uses axis=1 as "new_y")
    new_points[:, 1] -= bound_min[1]

    length = float(bound_max[0] - bound_min[0])
    if length <= 1e-12:
        raise RuntimeError(f"Degenerate length in x after transform: length={length}")

    scale = float(target_len / length)
    new_points *= scale

    new_points[:, 0] -= np.mean(new_points[:, 0])
    new_points[:, 2] -= np.mean(new_points[:, 2])

    return new_points, new_normals


def infer_sim_name_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    idx = base.find("sim_")
    if idx == -1:
        return ""
    cand = base[idx:idx + 7]
    if len(cand) == 7 and cand[:4] == "sim_" and cand[4:7].isdigit():
        return cand
    return ""


def collect_vtk_files(vtk_dir: str) -> list[str]:
    vtk_files = []
    for fn in os.listdir(vtk_dir):
        if fn.lower().endswith((".vtk", ".vtm", ".vtu", ".vtp")):
            vtk_files.append(fn)
    if not vtk_files:
        raise RuntimeError(f"No VTK-like files found in: {vtk_dir}")
    vtk_files.sort()
    return vtk_files


def process_split(vtk_dir: str, json_path: str, outdir: str, start_index: int, args) -> int:
    sim_map = build_sim_map(json_path)
    dtype = getattr(np, args.dtype)
    car_center = np.array(args.car_center, dtype=np.float64)

    vtk_files = collect_vtk_files(vtk_dir)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(vtk_files)

    index = int(start_index)

    for fn in vtk_files:
        vtk_path = os.path.join(vtk_dir, fn)

        vtk_path = os.path.abspath(str(vtk_path)).strip().rstrip("\x00")

        print(f"\n[Load] {vtk_path!r}")
        if not os.path.isfile(vtk_path):
            raise FileNotFoundError(f"Not a file: {vtk_path!r}")

        ext = os.path.splitext(vtk_path)[1].lower()
        if ext not in (".vtk", ".vtm", ".vtu", ".vtp"):
            raise RuntimeError(f"Bad extension parsed: ext={ext!r} from path={vtk_path!r}")

        mesh = read_hf_safe(vtk_path)

        if mesh.n_points == 0:
            raise RuntimeError(f"Loaded mesh has 0 points: {vtk_path}")

        surf = mesh.extract_surface()

        # scalar -> point_data (safe)
        surf = ensure_point_scalar(surf, args.scalar)

        # normals
        surf = surf.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=args.auto_orient_normals,
            flip_normals=args.flip_normals,
            inplace=False,
        )

        if args.normal_name not in surf.point_data:
            raise KeyError(
                f"Normal array '{args.normal_name}' not found. "
                f"Available point_data: {list(surf.point_data.keys())}"
            )

        xyz = np.asarray(surf.points, dtype=np.float64)
        nrm = np.asarray(surf.point_data[args.normal_name], dtype=np.float64)
        y_raw = np.asarray(surf.point_data[args.scalar], dtype=np.float64).reshape(-1, 1)

        sim_name = infer_sim_name_from_filename(fn)
        if sim_name == "":
            raise KeyError(
                f"Cannot infer sim name like 'sim_017' from filename: {fn}\n"
                f"Please rename files to include 'sim_XXX' or modify infer_sim_name_from_filename()."
            )
        if sim_name not in sim_map:
            raise KeyError(
                f"JSON has no entry for {sim_name}. Example keys: {list(sim_map.keys())[:5]} ..."
            )

        angle_deg = float(sim_map[sim_name]["rotation_angle_deg"])
        print(f"[Info] {sim_name} rotation_angle_deg = {angle_deg}")

        theta = np.deg2rad(-angle_deg)
        Rz = rotation_matrix_z(theta)

        pos_centered = xyz - car_center
        xyz_rot = pos_centered @ Rz.T + car_center

        nrm_rot = nrm @ Rz.T
        nrm_rot = nrm_rot / (np.linalg.norm(nrm_rot, axis=1, keepdims=True) + 1e-12)

        pts_new, nrm_new = transform_points_normals(xyz_rot, nrm_rot, target_len=args.target_len)
        x_out = np.concatenate([pts_new, nrm_new], axis=1)

        x_out = x_out.astype(dtype, copy=False)
        y_out = y_raw.astype(dtype, copy=False)
        cond = np.array([angle_deg], dtype=dtype)

        np.save(os.path.join(outdir, f"x_{index}.npy"), x_out)
        np.save(os.path.join(outdir, f"y_{index}.npy"), y_out)
        np.save(os.path.join(outdir, f"cond_{index}.npy"), cond)

        print(f"[OK] Saved index={index} x={x_out.shape} y={y_out.shape} c={cond[0]}")
        index += 1

    return index


def main():
    parser = argparse.ArgumentParser(
        description="VTK -> (surface xyz+normals, scalar) -> rotate/normalize -> save x_i/y_i/cond_i"
    )

    # ---------- HF options ----------
    parser.add_argument("--hf_repo", type=str, default="",
                        help="HF dataset repo id, e.g. HaixuWu/GeoPT (leave empty to use local dirs)")
    parser.add_argument("--hf_subdir", type=str, default="Car_Crash",
                        help="Subdir inside the HF repo to download, default Car_Crash")
    parser.add_argument("--hf_revision", type=str, default=None,
                        help="Optional HF revision/branch/tag/commit")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Optional HF cache dir (e.g. /data/NAS/hf_cache)")

    # ---------- Split dirs (local or resolved from HF) ----------
    parser.add_argument("--train_vtk_dir", type=str, default="Car_Crash/train")
    parser.add_argument("--test_vtk_dir", type=str, default="Car_Crash/test")
    parser.add_argument("--train_json", type=str, default="Car_Crash/simulations_info_train.json")
    parser.add_argument("--test_json", type=str, default="Car_Crash/simulations_info_test.json")

    parser.add_argument("--outdir", type=str, default="./",
                        help="Output folder to save x_i.npy, y_i.npy, cond_i.npy")

    parser.add_argument("--scalar", type=str, default="2DELEM_Von_Mises_Tmax")
    parser.add_argument("--normal_name", type=str, default="Normals")
    parser.add_argument("--flip_normals", action="store_true")
    parser.add_argument("--auto_orient_normals", action="store_true")

    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--test_start_index", type=int, default=97)

    parser.add_argument("--car_center", type=float, nargs=3,
                        default=[2494.2413886546033, 29.947353684824993, 601.6743326155072])
    parser.add_argument("--target_len", type=float, default=5.0)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- Resolve dirs from HF if requested ----------
    if args.hf_repo:
        local_root = hf_download_subdir(
            repo_id=args.hf_repo,
            subdir=args.hf_subdir,
            repo_type="dataset",
            cache_dir=args.hf_cache_dir,
            revision=args.hf_revision,
        )
        car_crash_root = os.path.join(local_root, args.hf_subdir)

        args.train_json = os.path.join(car_crash_root, "simulations_info_train.json")
        args.test_json = os.path.join(car_crash_root, "simulations_info_test.json")
        if not os.path.isfile(args.train_json):
            raise FileNotFoundError(f"train_json not found: {args.train_json}")
        if not os.path.isfile(args.test_json):
            raise FileNotFoundError(f"test_json not found: {args.test_json}")

        print("[HF] Train JSON:", args.train_json)
        print("[HF] Test  JSON:", args.test_json)

        # default layout assumptions: <root>/<subdir>/train and <root>/<subdir>/test
        hf_train = os.path.join(local_root, args.hf_subdir, "train")
        hf_test = os.path.join(local_root, args.hf_subdir, "test")

        # only overwrite if those exist; otherwise user can pass explicit dirs
        if os.path.isdir(hf_train):
            args.train_vtk_dir = hf_train
        if os.path.isdir(hf_test):
            args.test_vtk_dir = hf_test

        print("[HF] Local root:", local_root)
        print("[HF] Train dir :", args.train_vtk_dir)
        print("[HF] Test dir  :", args.test_vtk_dir)

    # ---------- Run train then test ----------
    print("\n=== Process TRAIN ===")
    process_split(args.train_vtk_dir, args.train_json, args.outdir, args.train_start_index, args)

    print("\n=== Process TEST ===")
    process_split(args.test_vtk_dir, args.test_json, args.outdir, args.test_start_index, args)


if __name__ == "__main__":
    main()
