#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NASA CRM (HDF5) -> preprocess -> save x_i / y_i / cond_i

This script MERGES the functionality of the three provided scripts and
keeps the SAME STRUCTURE and OUTPUT FORMAT as your previous AirCraft pipeline.

Input:
- trainingData_NASA-CRM.h5
- testData_NASA-CRM.h5

Output (separated by split):
outdir/
  train/
    x_1.npy, y_1.npy, cond_1.npy
    ...
  test/
    x_1.npy, y_1.npy, cond_1.npy

Data conventions:
- x: (N, 7) = [x, y, z, 0, nx, ny, nz]
- y: (N,)   = PressureCoefficient
- cond:
    default: [Mach, AlphaMean]
    optional (--cond_full): full 6 control parameters
"""

import argparse
import os
import numpy as np
import h5py


# ----------------------------------------------------------------------
# Build input feature x from a single HDF5 group (one sample)
# ----------------------------------------------------------------------
def build_x_from_h5_group(group: h5py.Group) -> np.ndarray:
    """
    Assemble surface input features:
      x = [CoordinateX, CoordinateY, CoordinateZ, 0, NormalX, NormalY, NormalZ]

    Shape: (N, 7)
    """
    cx = group["CoordinateX"][:]
    cy = group["CoordinateY"][:]
    cz = group["CoordinateZ"][:]

    nx = group["NormalX"][:]
    ny = group["NormalY"][:]
    nz = group["NormalZ"][:]

    zeros = np.zeros_like(cx)

    x = np.stack([cx, cy, cz, zeros, nx, ny, nz], axis=-1)
    return x.astype(np.float64, copy=False)


# ----------------------------------------------------------------------
# Geometry transform
# ----------------------------------------------------------------------
def transform_like_file3(x: np.ndarray, target_len: float = 5.0) -> np.ndarray:
    """
    Apply the exact geometric transform used in the original script:

    - Swap Y/Z coordinates
    - Swap Y/Z components of normals
    - Shift Y so that min(Y) = 0
    - Scale geometry so that X-extent == target_len
    - Center X around zero

    Input / Output shape: (N, 7)
    """
    x = np.asarray(x, dtype=np.float64)
    new_x = np.zeros_like(x, dtype=np.float64)

    # --- positions ---
    new_x[:, 0] = x[:, 0]   # X
    new_x[:, 1] = x[:, 2]   # Z -> Y
    new_x[:, 2] = x[:, 1]   # Y -> Z
    new_x[:, 3] = x[:, 3]   # zero column

    # --- normals ---
    new_x[:, 4] = x[:, 4]   # nx
    new_x[:, 5] = x[:, 6]   # nz -> ny
    new_x[:, 6] = x[:, 5]   # ny -> nz

    bound_max = np.max(new_x, axis=0)
    bound_min = np.min(new_x, axis=0)

    # Shift Y so that the minimum is zero
    new_x[:, 1] -= bound_min[1]

    # Scale using X-extent
    length = float(bound_max[0] - bound_min[0])
    if length <= 1e-12:
        raise RuntimeError("Degenerate geometry: zero X-extent")

    scale = target_len / length
    new_x[:, :3] *= scale

    # Center X
    new_x[:, 0] -= np.mean(new_x[:, 0])

    return new_x


# ----------------------------------------------------------------------
# Extract condition vector from sample attributes
# ----------------------------------------------------------------------
def parse_condition(group: h5py.Group, full: bool = False) -> np.ndarray:
    """
    Build conditioning vector from HDF5 attributes.

    Default:
      cond = [Mach, AlphaMean]

    If full=True:
      cond = [Mach, AlphaMean, aileronInboard,
              aileronOutboard, elevator, htp]
    """
    if not full:
        return np.array(
            [group.attrs["Mach"], group.attrs["AlphaMean"]],
            dtype=np.float32,
        )

    return np.array(
        [
            group.attrs["Mach"],
            group.attrs["AlphaMean"],
            group.attrs["aileronInboard"],
            group.attrs["aileronOutboard"],
            group.attrs["elevator"],
            group.attrs["htp"],
        ],
        dtype=np.float32,
    )


# ----------------------------------------------------------------------
# Process one split (train or test)
# ----------------------------------------------------------------------
def process_split(
    h5_path: str,
    split: str,
    outdir: str,
    start_index: int,
    target_len: float,
    dtype: str,
    cond_full: bool,
    skip_existing: bool,
):
    """
    Convert all samples in one HDF5 file into x_i / y_i / cond_i.
    """
    os.makedirs(os.path.join(outdir, split), exist_ok=True)
    np_dtype = getattr(np, dtype)

    print(f"[Load] {split}: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # Use sorted keys for deterministic ordering
        samples = sorted(f.keys())
        print(f"[Info] {split} samples: {len(samples)}")

        idx = start_index
        for sample_name in samples:
            group = f[sample_name]

            x_path = os.path.join(outdir, split, f"x_{idx}.npy")
            y_path = os.path.join(outdir, split, f"y_{idx}.npy")
            c_path = os.path.join(outdir, split, f"cond_{idx}.npy")

            if skip_existing and all(map(os.path.exists, [x_path, y_path, c_path])):
                print(f"[Skip] {split} index={idx} already exists")
                idx += 1
                continue

            # Build input features
            x = build_x_from_h5_group(group)
            x = transform_like_file3(x, target_len=target_len)
            x = x.astype(np_dtype, copy=False)

            # Output field
            y = np.asarray(group["PressureCoefficient"][:], dtype=np_dtype)

            # Condition
            cond = parse_condition(group, full=cond_full).astype(np_dtype, copy=False)

            # Save
            np.save(x_path, x)
            np.save(y_path, y)
            np.save(c_path, cond)

            print(
                f"[OK] {split} {sample_name} -> "
                f"idx={idx}, x={x.shape}, y={y.shape}, cond={cond.tolist()}"
            )
            idx += 1


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NASA CRM HDF5 -> preprocess -> save x_i/y_i/cond_i"
    )

    parser.add_argument(
        "--train_h5",
        type=str,
        default="./trainingData_NASA-CRM.h5",
        help="Training HDF5 file",
    )
    parser.add_argument(
        "--test_h5",
        type=str,
        default="./testData_NASA-CRM.h5",
        help="Test HDF5 file",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./nasa_full_size",
        help="Output directory",
    )

    parser.add_argument("--train_start_index", type=int, default=1)
    parser.add_argument("--test_start_index", type=int, default=1)

    parser.add_argument("--target_len", type=float, default=5.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
    )

    parser.add_argument(
        "--cond_full",
        action="store_true",
        help="Use full 6-dimensional control vector for cond",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already exist on disk",
    )

    args = parser.parse_args()

    process_split(
        h5_path=args.train_h5,
        split="train",
        outdir=args.outdir,
        start_index=args.train_start_index,
        target_len=args.target_len,
        dtype=args.dtype,
        cond_full=args.cond_full,
        skip_existing=args.skip_existing,
    )

    process_split(
        h5_path=args.test_h5,
        split="test",
        outdir=args.outdir,
        start_index=args.test_start_index,
        target_len=args.target_len,
        dtype=args.dtype,
        cond_full=args.cond_full,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
