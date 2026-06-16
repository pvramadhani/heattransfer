import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .data import NPZFieldDataset
from .model import build_generator, build_discriminator
from .losses import (
    adversarial_generator_loss,
    discriminator_loss,
    l1_temperature_loss,
    gradient_consistency_loss,
    boundary_loss,
    laplacian_residual_loss,
)
from .metrics import ssim_2d


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=2)

    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)

    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--lambda_grad", type=float, default=10.0)
    p.add_argument("--lambda_bc", type=float, default=20.0)
    p.add_argument("--lambda_residual", type=float, default=0.0)

    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_obj = NPZFieldDataset(
        args.data,
        val_split=args.val_split,
        seed=args.seed,
    )

    train_ds = ds_obj.tf_dataset(
        "train",
        args.batch,
        shuffle=True,
    )

    val_ds = ds_obj.tf_dataset(
        "val",
        args.batch,
        shuffle=False,
    )

    input_shape = ds_obj.x.shape[1:]

    G = build_generator(input_shape, dim=args.dim)
    D = build_discriminator(input_shape, dim=args.dim)

    opt_g = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5)
    opt_d = tf.keras.optimizers.Adam(args.lr_d, beta_1=0.5)

    @tf.function
    def train_step(x, y, bmask):
        with tf.GradientTape() as tg, tf.GradientTape() as td:
            y_fake = G(x, training=True)

            real_logits = D([x, y], training=True)
            fake_logits = D([x, y_fake], training=True)

            d_loss = discriminator_loss(real_logits, fake_logits)

            adv = adversarial_generator_loss(fake_logits)
            l1 = l1_temperature_loss(y, y_fake)
            grad = gradient_consistency_loss(y, y_fake, args.dim)
            bc = boundary_loss(y, y_fake, bmask)
            res = laplacian_residual_loss(y_fake, None, args.dim)

            g_loss = (
                adv
                + args.lambda_l1 * l1
                + args.lambda_grad * grad
                + args.lambda_bc * bc
                + args.lambda_residual * res
            )

        g_grads = tg.gradient(g_loss, G.trainable_variables)
        d_grads = td.gradient(d_loss, D.trainable_variables)

        opt_g.apply_gradients(zip(g_grads, G.trainable_variables))
        opt_d.apply_gradients(zip(d_grads, D.trainable_variables))

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "adv": adv,
            "l1": l1,
            "grad": grad,
            "bc": bc,
            "res": res,
        }

    history = []

    for epoch in range(1, args.epochs + 1):
        logs = []

        for x, y, b in tqdm(train_ds, desc=f"epoch {epoch}/{args.epochs}"):
            step_log = train_step(x, y, b)
            logs.append({k: float(v.numpy()) for k, v in step_log.items()})

        row = {"epoch": epoch}

        for k in logs[0]:
            row[k] = float(np.mean([m[k] for m in logs]))

        val_l1, val_ssim = [], []

        for x, y, b in val_ds:
            pred = G(x, training=False)
            val_l1.append(float(tf.reduce_mean(tf.abs(y - pred)).numpy()))

            if args.dim == 2:
                val_ssim.append(float(ssim_2d(y, pred).numpy()))

        row["val_l1_norm"] = float(np.mean(val_l1)) if val_l1 else None
        row["val_ssim"] = float(np.mean(val_ssim)) if val_ssim else None

        print(json.dumps(row, indent=2))
        history.append(row)

        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            G.save(out / "generator.keras")
            D.save(out / "discriminator.keras")

            np.savez(
                out / "normalizer.npz",
                x_min=ds_obj.norm.x_min,
                x_max=ds_obj.norm.x_max,
                y_min=ds_obj.norm.y_min,
                y_max=ds_obj.norm.y_max,
            )

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
