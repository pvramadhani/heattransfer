"""Prepare the uploaded CFD dataset for PI-cGAN training.

Expected dataset structure after extraction:
    datasets/BC_Mask
    datasets/Simulation Results/dataset_Q105_T10_h700.txt

The converter is intentionally tolerant to common CFD text exports:
- 2D matrix values: H x W
- flattened vector: H*W
- point table: x y value, or x y z value; the last column is treated as field value
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PATTERN = re.compile(
    r"dataset_Q(?P<Q>[-+0-9.]+)_T(?P<T>[-+0-9.]+)_h(?P<h>[-+0-9.]+)\.txt$",
    re.I,
)


def find_extractor() -> Optional[str]:
    for exe in ("unrar", "unar", "7z", "7zz", "7za"):
        path = shutil.which(exe)
        if path:
            return path
    return None


def extract_rar(rar_path: Path, out_dir: Path) -> Path:
    exe = find_extractor()
    if exe is None:
        raise RuntimeError(
            "Tidak menemukan extractor RAR. Install salah satu: unrar, unar, 7z; "
            "atau extract manual lalu gunakan --root <folder_datasets>."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(exe).name.lower()

    if name.startswith("unrar"):
        cmd = [exe, "x", "-o+", str(rar_path), str(out_dir)]
    elif name == "unar":
        cmd = [exe, "-o", str(out_dir), str(rar_path)]
    else:
        cmd = [exe, "x", f"-o{out_dir}", str(rar_path)]

    subprocess.run(cmd, check=True)

    candidates = [out_dir / "datasets"] + [
        p for p in out_dir.rglob("datasets") if p.is_dir()
    ]

    for c in candidates:
        if (c / "Simulation Results").exists() and (c / "BC_Mask").exists():
            return c

    raise FileNotFoundError(
        "Folder datasets/ dengan BC_Mask dan Simulation Results tidak ditemukan."
    )


def load_numeric(path: Path) -> np.ndarray:
    """Load numeric txt/csv-like CFD export."""
    last_err = None

    for delimiter in (None, ",", ";", "\t"):
        try:
            arr = np.loadtxt(
                path,
                delimiter=delimiter,
                comments=("#", "%"),
                dtype=np.float32,
            )
            if arr.size == 0:
                raise ValueError("empty file")
            return arr
        except Exception as e:
            last_err = e

    raise ValueError(f"Gagal membaca file numerik: {path}. Error terakhir: {last_err}")


def table_to_grid(arr: np.ndarray) -> np.ndarray:
    """Convert common CFD point tables to a 2D grid.

    If arr has 3 columns, assumes x y value.
    If arr has 4 or 5 columns, uses first two columns as x-y coordinates
    and the last column as field value.
    """
    if arr.ndim == 1:
        return arr

    if arr.ndim == 2 and arr.shape[1] in (3, 4, 5):
        x = arr[:, 0]
        y = arr[:, 1]
        v = arr[:, -1]

        ux = np.unique(x)
        uy = np.unique(y)

        if ux.size * uy.size == v.size:
            ix = {val: i for i, val in enumerate(np.sort(ux))}
            iy = {val: i for i, val in enumerate(np.sort(uy))}

            grid = np.empty((uy.size, ux.size), dtype=np.float32)

            for xx, yy, vv in zip(x, y, v):
                grid[iy[yy], ix[xx]] = vv

            return grid

    return arr


def infer_shape_from_mask(
    mask: np.ndarray,
    user_shape: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    if user_shape is not None:
        return user_shape

    if mask.ndim == 2 and min(mask.shape) > 1:
        return int(mask.shape[0]), int(mask.shape[1])

    n = int(mask.size)

    factors = []
    for h in range(8, int(np.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            factors.append((h, w, abs((w / h) - 2.0)))

    if not factors:
        raise ValueError(
            f"Tidak bisa infer shape dari {n} elemen. Gunakan --shape H W."
        )

    h, w, _ = min(factors, key=lambda t: t[2])
    return h, w


def as_grid(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    arr = table_to_grid(arr)

    if arr.ndim == 2 and arr.shape == shape:
        return arr.astype(np.float32)

    flat = arr.reshape(-1)

    if flat.size != shape[0] * shape[1]:
        raise ValueError(
            f"Jumlah elemen {flat.size} tidak cocok dengan shape {shape}."
        )

    return flat.reshape(shape).astype(np.float32)


def coordinate_channels(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    return np.broadcast_to(xx, (h, w)), np.broadcast_to(yy, (h, w))


def build_condition(mask: np.ndarray, Q: float, T: float, hcoef: float) -> np.ndarray:
    H, W = mask.shape
    xcoord, ycoord = coordinate_channels((H, W))

    channels = [
        mask.astype(np.float32),
        np.full((H, W), Q, dtype=np.float32),
        np.full((H, W), T, dtype=np.float32),
        np.full((H, W), hcoef, dtype=np.float32),
        xcoord.astype(np.float32),
        ycoord.astype(np.float32),
    ]

    return np.stack(channels, axis=-1)


def collect_files(root: Path) -> List[Tuple[Path, float, float, float]]:
    sim_dir = root / "Simulation Results"

    if not sim_dir.exists():
        raise FileNotFoundError(f"Tidak menemukan folder: {sim_dir}")

    items = []

    for p in sim_dir.glob("dataset_Q*_T*_h*.txt"):
        m = PATTERN.search(p.name)
        if not m:
            continue

        items.append(
            (
                p,
                float(m.group("Q")),
                float(m.group("T")),
                float(m.group("h")),
            )
        )

    if not items:
        raise FileNotFoundError(
            f"Tidak ada file dataset_Q*_T*_h*.txt di {sim_dir}"
        )

    return sorted(items, key=lambda z: (z[1], z[2], z[3]))


def convert(
    root: Path,
    out: Path,
    shape: Optional[Tuple[int, int]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    mask_raw = load_numeric(root / "BC_Mask")
    mask_grid0 = table_to_grid(mask_raw)
    grid_shape = infer_shape_from_mask(mask_grid0, shape)

    mask = as_grid(mask_raw, grid_shape)
    boundary = (np.abs(mask) > 0).astype(np.float32)[..., None]

    files = collect_files(root)

    if max_samples:
        files = files[:max_samples]

    xs, ys, params = [], [], []

    for p, Q, T, hcoef in files:
        field = as_grid(load_numeric(p), grid_shape)

        xs.append(build_condition(mask, Q, T, hcoef))
        ys.append(field[..., None])
        params.append([Q, T, hcoef])

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.stack(ys, axis=0).astype(np.float32)
    b = np.broadcast_to(boundary, y.shape).astype(np.float32)

    params_arr = np.array(params, dtype=np.float32)

    param_names = np.array(["Q", "T", "h"], dtype=object)
    channel_names = np.array(
        ["bc_mask", "Q_field", "T_field", "h_field", "x_coord", "y_coord"],
        dtype=object,
    )

    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        x=x,
        y=y,
        boundary=b,
        params=params_arr,
        param_names=param_names,
        channel_names=channel_names,
    )

    meta = {
        "out": str(out),
        "n_samples": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "grid_shape": list(grid_shape),
        "params_min": params_arr.min(axis=0).tolist(),
        "params_max": params_arr.max(axis=0).tolist(),
        "param_names": param_names.tolist(),
        "channel_names": channel_names.tolist(),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }

    out.with_suffix(".metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return meta


def main() -> None:
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rar", type=Path, help="Path ke datasets.rar")
    src.add_argument("--root", type=Path, help="Path ke folder datasets hasil extract")

    ap.add_argument("--out", type=Path, default=Path("data/vc_real_dataset.npz"))
    ap.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"), default=None)
    ap.add_argument("--max_samples", type=int, default=None)

    args = ap.parse_args()

    if args.rar:
        with tempfile.TemporaryDirectory(prefix="vc_dataset_") as td:
            root = extract_rar(args.rar, Path(td))
            meta = convert(
                root,
                args.out,
                tuple(args.shape) if args.shape else None,
                args.max_samples,
            )
    else:
        meta = convert(
            args.root,
            args.out,
            tuple(args.shape) if args.shape else None,
            args.max_samples,
        )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
