#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import glob
import numpy as np
import h5py
from typing import Optional


# ---------------- HF download ----------------
def hf_download_subdir(
    repo_id: str,
    subdir: str,
    repo_type: str = "dataset",
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Download only a subdir from a Hugging Face repo into local cache and return local snapshot root.
    """
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


# ---------------- Preprocess logic ----------------
def transform(pos: np.ndarray, normal: np.ndarray, target_length: float = 5.0) -> np.ndarray:
    """
    Keep your original transform behavior:
      - concatenate pos + normal into (N,6)
      - scale coords by target_length / x_extent
      - center x by subtracting mean
      - shift y by -min(y)
      - center z by subtracting mid of min/max z (from original bounds)
    """
    pos = np.asarray(pos, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)

    new_data = np.zeros((pos.shape[0], 6), dtype=np.float64)
    new_data[:, 0:3] = pos[:, 0:3]
    new_data[:, 3:6] = normal[:, 0:3]

    bound_max = np.max(new_data, axis=0)
    bound_min = np.min(new_data, axis=0)

    length = float(bound_max[0] - bound_min[0])  # x-extent
    if length <= 1e-12:
        raise RuntimeError(f"Degenerate x-extent: {length}")

    scale = float(target_length / length)
    new_data[:, :3] *= scale

    x_avg = float(np.mean(new_data[:, 0]))
    new_data[:, 0] -= x_avg

    # IMPORTANT: your original used bound_min from *pre-scale* bounds for y/z shift.
    # To preserve exact behavior, we keep it. If you want fully consistent geometry,
    # change to use scaled bounds.
    new_data[:, 1] -= float(bound_min[1])
    new_data[:, 2] -= float((bound_min[2] + bound_max[2]) / 2.0)

    return new_data


def collect_h5_paths(root_dir: str, pattern: str) -> list:
    """
    Recursively find .h5 files under root_dir, then apply basename glob pattern.
    Example pattern: "*0.h5"
    """
    all_h5 = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".h5"):
                all_h5.append(os.path.join(r, fn))

    # apply pattern on basename
    import fnmatch
    matched = [p for p in all_h5 if fnmatch.fnmatch(os.path.basename(p), pattern)]
    matched.sort()
    return matched


def process_h5_files(h5_paths: list, outdir: str, start_index: int, args) -> None:
    os.makedirs(outdir, exist_ok=True)
    dtype = getattr(np, args.dtype)

    paths = list(h5_paths)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(paths)

    file_list = []
    index = int(start_index)

    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]  # remove .h5
        file_list.append(base)

        print(f"\n[Load] {p}")

        with h5py.File(p, "r") as f:
            # safer: materialize arrays
            pos = f[args.pos_key][:]
            normals = f[args.normals_key][:]

            new_data = transform(pos, normals, target_length=args.target_len)  # (N,6)

            # match your original output shape: (N,7) = xyz + 0 + normals
            new_data = np.concatenate(
                (new_data[:, :3], np.zeros((new_data.shape[0], 1), dtype=np.float64), new_data[:, 3:]),
                axis=1
            )

            y = f[args.values_key][:]
            # ensure y is (N,1) or (N,) consistent; don't force reshape unless you want
            # y = np.asarray(y).reshape(-1, 1)

        x_out = new_data.astype(dtype, copy=False)
        y_out = np.asarray(y, dtype=dtype)

        # If no condition exists, store dummy cond=0.0 (or store target_len, etc.)
        cond = np.array([float(base.split('_')[1]), float(base.split('_')[2]), float(base.split('_')[3])])

        np.save(os.path.join(outdir, f"x_{index}.npy"), x_out)
        np.save(os.path.join(outdir, f"y_{index}.npy"), y_out)
        np.save(os.path.join(outdir, f"cond_{index}.npy"), cond)

        print(f"[OK] Saved index={index} x={x_out.shape} y={y_out.shape} cond={cond.shape}")
        index += 1


def main():
    parser = argparse.ArgumentParser(description="GeoPT/AirCraft H5 -> preprocess -> save x_i/y_i/cond_i")

    # ---- HF options ----
    parser.add_argument("--hf_repo", type=str, default="HaixuWu/GeoPT",
                        help="HF dataset repo id, e.g. HaixuWu/GeoPT")
    parser.add_argument("--hf_subdir", type=str, default="AirCraft",
                        help="Subdir inside repo to download, e.g. AirCraft")
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Optional HF cache dir, e.g. ./hf_cache")

    # ---- local override ----
    parser.add_argument("--h5_dir", type=str, default="",
                        help="If set, skip HF and read h5 from this local dir")

    # ---- IO ----
    parser.add_argument("--outdir", type=str, default="./aircraft_full_size",
                        help="Output dir for x_i/y_i/cond_i + data_list.npy")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--pattern", type=str, default="*0.h5",
                        help="Basename glob pattern to match, default '*0.h5'")

    # ---- H5 keys ----
    parser.add_argument("--pos_key", type=str, default="pos")
    parser.add_argument("--normals_key", type=str, default="normals")
    parser.add_argument("--values_key", type=str, default="values")

    # ---- preprocess params ----
    parser.add_argument("--target_len", type=float, default=5.0,
                        help="scale = target_len / x_extent")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])

    # ---- order ----
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Fix duplicated arg issue if you copy-pasted: ensure only one target_len
    # (If you keep the script as-is, delete the duplicate parser.add_argument lines.)
    # Here we rely on args.target_len.

    # Resolve h5_dir
    if args.h5_dir:
        h5_root = args.h5_dir
        print("[Local] h5_dir:", h5_root)
    else:
        local_root = hf_download_subdir(
            repo_id=args.hf_repo,
            subdir=args.hf_subdir,
            repo_type="dataset",
            cache_dir=args.hf_cache_dir,
            revision=args.hf_revision,
        )
        h5_root = os.path.join(local_root, args.hf_subdir)
        print("[HF] Local root:", local_root)
        print("[HF] h5_root  :", h5_root)

    # Collect files
    h5_paths = collect_h5_paths(h5_root, args.pattern)
    if not h5_paths:
        raise RuntimeError(f"No h5 files matched pattern={args.pattern!r} under {h5_root}")

    print(f"[Info] matched {len(h5_paths)} files. First 5:")
    for p in h5_paths[:5]:
        print(" ", p)

    process_h5_files(h5_paths, args.outdir, args.start_index, args)


if __name__ == "__main__":
    main()
