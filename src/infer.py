import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .data import NPZFieldDataset


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--max_samples", type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = NPZFieldDataset(
        args.data,
        val_split=0.2,
        seed=42,
    )

    x, y, b = ds.arrays("val")

    G = tf.keras.models.load_model(
        args.ckpt,
        compile=False,
    )

    pred = G.predict(
        x[: args.max_samples],
        batch_size=1,
    )

    pred_k = ds.norm.inverse_y(pred)
    true_k = ds.norm.inverse_y(y[: args.max_samples])

    np.savez(
        out / "predictions.npz",
        pred=pred_k,
        true=true_k,
        x=x[: args.max_samples],
    )

    if args.dim == 2:
        for i in range(min(args.max_samples, len(pred_k))):
            fig, ax = plt.subplots(1, 3, figsize=(12, 3))

            im0 = ax[0].imshow(true_k[i, ..., 0])
            ax[0].set_title("CFD/true K")
            plt.colorbar(im0, ax=ax[0])

            im1 = ax[1].imshow(pred_k[i, ..., 0])
            ax[1].set_title("cGAN pred K")
            plt.colorbar(im1, ax=ax[1])

            im2 = ax[2].imshow(
                np.abs(true_k[i, ..., 0] - pred_k[i, ..., 0])
            )
            ax[2].set_title("abs error K")
            plt.colorbar(im2, ax=ax[2])

            for a in ax:
                a.axis("off")

            fig.tight_layout()
            fig.savefig(out / f"sample_{i}.png", dpi=200)
            plt.close(fig)

    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
