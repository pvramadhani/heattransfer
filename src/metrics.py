import tensorflow as tf


def mse_kelvin(y_true_k, y_pred_k):
    return tf.reduce_mean(tf.square(y_true_k - y_pred_k))


def mae_kelvin(y_true_k, y_pred_k):
    return tf.reduce_mean(tf.abs(y_true_k - y_pred_k))


def max_abs_error_kelvin(y_true_k, y_pred_k):
    return tf.reduce_max(tf.abs(y_true_k - y_pred_k))


def ssim_2d(y_true_norm, y_pred_norm):
    """SSIM for 2D fields only.

    Inputs are normalized to [-1, 1].
    """
    return tf.reduce_mean(
        tf.image.ssim(
            (y_true_norm + 1) / 2,
            (y_pred_norm + 1) / 2,
            max_val=1.0,
        )
    )
