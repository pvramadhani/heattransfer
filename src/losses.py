import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def adversarial_generator_loss(fake_logits):
    return bce(tf.ones_like(fake_logits), fake_logits)


def discriminator_loss(real_logits, fake_logits):
    real = bce(tf.ones_like(real_logits), real_logits)
    fake = bce(tf.zeros_like(fake_logits), fake_logits)

    return 0.5 * (real + fake)


def l1_temperature_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def finite_gradients(u, dim=2):
    if dim == 2:
        dy = u[:, 1:, :, :] - u[:, :-1, :, :]
        dx = u[:, :, 1:, :] - u[:, :, :-1, :]

        return dx, dy

    dz = u[:, 1:, :, :, :] - u[:, :-1, :, :, :]
    dy = u[:, :, 1:, :, :] - u[:, :, :-1, :, :]
    dx = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]

    return dx, dy, dz


def gradient_consistency_loss(y_true, y_pred, dim=2):
    gt = finite_gradients(y_true, dim)
    gp = finite_gradients(y_pred, dim)

    return tf.add_n(
        [tf.reduce_mean(tf.square(a - b)) for a, b in zip(gt, gp)]
    ) / float(len(gt))


def boundary_loss(y_true, y_pred, boundary_mask):
    denom = tf.reduce_sum(boundary_mask) + 1e-6

    return tf.reduce_sum(
        tf.abs(y_true - y_pred) * boundary_mask
    ) / denom


def laplacian_residual_loss(y_pred, domain_mask=None, dim=2):
    """Optional steady-conduction residual.

    This approximates Laplacian(T)=0 away from explicit source terms.
    For more advanced physics-informed training, this can be extended to:
        div(k_eff grad T) + S_E = 0
    """
    if dim == 2:
        center = y_pred[:, 1:-1, 1:-1, :]

        lap = (
            y_pred[:, 2:, 1:-1, :]
            + y_pred[:, :-2, 1:-1, :]
            + y_pred[:, 1:-1, 2:, :]
            + y_pred[:, 1:-1, :-2, :]
            - 4.0 * center
        )

        if domain_mask is not None:
            mask = domain_mask[:, 1:-1, 1:-1, :]

            return tf.reduce_sum(tf.square(lap) * mask) / (
                tf.reduce_sum(mask) + 1e-6
            )

        return tf.reduce_mean(tf.square(lap))

    center = y_pred[:, 1:-1, 1:-1, 1:-1, :]

    lap = (
        y_pred[:, 2:, 1:-1, 1:-1, :]
        + y_pred[:, :-2, 1:-1, 1:-1, :]
        + y_pred[:, 1:-1, 2:, 1:-1, :]
        + y_pred[:, 1:-1, :-2, 1:-1, :]
        + y_pred[:, 1:-1, 1:-1, 2:, :]
        + y_pred[:, 1:-1, 1:-1, :-2, :]
        - 6.0 * center
    )

    if domain_mask is not None:
        mask = domain_mask[:, 1:-1, 1:-1, 1:-1, :]

        return tf.reduce_sum(tf.square(lap) * mask) / (
            tf.reduce_sum(mask) + 1e-6
        )

    return tf.reduce_mean(tf.square(lap))
