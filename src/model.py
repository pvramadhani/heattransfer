import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv(dim):
    return layers.Conv3D if dim == 3 else layers.Conv2D


def _deconv(dim):
    return layers.Conv3DTranspose if dim == 3 else layers.Conv2DTranspose


def conv_block(x, filters, dim=2, stride=2, norm=True, name=None):
    Conv = _conv(dim)

    x = Conv(
        filters,
        4,
        strides=stride,
        padding="same",
        use_bias=not norm,
        name=None if name is None else name + "_conv",
    )(x)

    if norm:
        x = layers.BatchNormalization(
            name=None if name is None else name + "_bn"
        )(x)

    return layers.LeakyReLU(
        0.2,
        name=None if name is None else name + "_lrelu",
    )(x)


def up_block(x, skip, filters, dim=2, dropout=False, name=None):
    Deconv = _deconv(dim)

    x = Deconv(
        filters,
        4,
        strides=2,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_deconv",
    )(x)

    x = layers.BatchNormalization(
        name=None if name is None else name + "_bn"
    )(x)

    if dropout:
        x = layers.Dropout(
            0.5,
            name=None if name is None else name + "_drop",
        )(x)

    x = layers.ReLU(name=None if name is None else name + "_relu")(x)

    if skip is not None:
        x = layers.Concatenate(
            name=None if name is None else name + "_cat"
        )([x, skip])

    return x


def build_generator(input_shape, dim=2, base=32):
    inp = layers.Input(shape=input_shape, name="condition")

    e1 = conv_block(inp, base, dim, norm=False, name="enc1")
    e2 = conv_block(e1, base * 2, dim, name="enc2")
    e3 = conv_block(e2, base * 4, dim, name="enc3")
    e4 = conv_block(e3, base * 8, dim, name="enc4")

    b = conv_block(e4, base * 8, dim, name="bottleneck")

    d1 = up_block(b, e4, base * 8, dim, dropout=True, name="dec1")
    d2 = up_block(d1, e3, base * 4, dim, name="dec2")
    d3 = up_block(d2, e2, base * 2, dim, name="dec3")
    d4 = up_block(d3, e1, base, dim, name="dec4")

    Deconv = _deconv(dim)

    out = Deconv(
        1,
        4,
        strides=2,
        padding="same",
        activation="tanh",
        name="temperature_norm",
    )(d4)

    return Model(inp, out, name="PI_cGAN_Generator")


def build_discriminator(input_shape, dim=2, base=32):
    cond = layers.Input(shape=input_shape, name="condition")
    temp = layers.Input(shape=input_shape[:-1] + (1,), name="temperature")

    x = layers.Concatenate()([cond, temp])

    x = conv_block(x, base, dim, norm=False, name="patch1")
    x = conv_block(x, base * 2, dim, name="patch2")
    x = conv_block(x, base * 4, dim, name="patch3")

    Conv = _conv(dim)

    x = Conv(
        base * 8,
        4,
        strides=1,
        padding="same",
        use_bias=False,
        name="patch4_conv",
    )(x)

    x = layers.BatchNormalization(name="patch4_bn")(x)
    x = layers.LeakyReLU(0.2)(x)

    out = Conv(
        1,
        4,
        strides=1,
        padding="same",
        name="patch_logits",
    )(x)

    return Model([cond, temp], out, name="Conditional_PatchGAN")
