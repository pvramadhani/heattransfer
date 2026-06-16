"""Simple surrogate-based operating-condition optimization for the real dataset.

For the uploaded dataset, the condition channels are:
0 bc_mask, 1 Q_field, 2 T_field, 3 h_field, 4 x_coord, 5 y_coord.

The objective minimizes predicted Tmax plus a thermal-spread penalty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import differential_evolution

from .data import NPZFieldDataset

REAL_CHANNELS = {
    "Q": 1,
    "T": 2,
    "h": 3,
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="NPZ hasil src.prepare_dataset")
    p.add_argument("--out", required=True)

    p.add_argument("--sample", type=int, default=0)

    p.add_argument("--q_bounds", type=float, nargs=2, default=(30.0, 150.0))
    p.add_argument("--T_bounds", type=float, nargs=2, default=(10.0, 25.0))
    p.add_argument("--h_bounds", type=float, nargs=2, default=(700.0, 1200.0))

    p.add_argument("--alpha_spread", type=float, default=0.25)
    p.add_argument("--target", choices=["min_tmax", "min_spread"], default="min_tmax")

    return p.parse_args()


def overwrite_condition(
    x_raw: np.ndarray,
    Q: float,
    T: float,
    hcoef: float,
) -> np.ndarray:
    x = x_raw.copy()

    x[..., REAL_CHANNELS["Q"]] = Q
    x[..., REAL_CHANNELS["T"]] = T
    x[..., REAL_CHANNELS["h"]] = hcoef

    return x


def main() -> None:
    args = parse_args()

    ds = NPZFieldDataset(
        args.data,
        val_split=0.2,
        seed=42,
    )

    if args.sample >= len(ds.x):
        raise IndexError(f"sample={args.sample} di luar jumlah data {len(ds.x)}")

    x_template = ds.x[args.sample : args.sample + 1]

    G = tf.keras.models.load_model(
        args.ckpt,
        compile=False,
    )

    def predict_stats(v):
        Q, T, hcoef = map(float, v)

        x_raw = overwrite_condition(
            x_template,
            Q,
            T,
            hcoef,
        )

        x_norm = ds.norm.transform_x(x_raw).astype(np.float32)

        pred_norm = G(
            x_norm,
            training=False,
        ).numpy()

        pred_k = ds.norm.inverse_y(pred_norm)

        tmax = float(pred_k.max())
        tmin = float(pred_k.min())
        tmean = float(pred_k.mean())
        spread = tmax - tmin

        return tmax, tmin, tmean, spread

    def objective(v):
        tmax, _tmin, _tmean, spread = predict_stats(v)

        if args.target == "min_spread":
            return spread

        return tmax + args.alpha_spread * spread

    bounds = [
        tuple(args.q_bounds),
        tuple(args.T_bounds),
        tuple(args.h_bounds),
    ]

    res = differential_evolution(
        objective,
        bounds,
        seed=7,
        polish=True,
        workers=1,
    )

    tmax, tmin, tmean, spread = predict_stats(res.x)

    result = {
        "objective": float(res.fun),
        "best_params": {
            "Q": float(res.x[0]),
            "T": float(res.x[1]),
            "h": float(res.x[2]),
        },
        "predicted_stats": {
            "Tmax": tmax,
            "Tmin": tmin,
            "Tmean": tmean,
            "spread": spread,
        },
        "bounds": {
            "Q": list(args.q_bounds),
            "T": list(args.T_bounds),
            "h": list(args.h_bounds),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
