import numpy as np


class Normalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def fit(self, x, y):
        axes_x = tuple(range(x.ndim - 1))

        self.x_min = x.min(axis=axes_x, keepdims=True).astype(np.float32)
        self.x_max = x.max(axis=axes_x, keepdims=True).astype(np.float32)

        self.y_min = np.array(y.min(), dtype=np.float32)
        self.y_max = np.array(y.max(), dtype=np.float32)

        return self

    def transform_x(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min + self.eps)

    def transform_y(self, y):
        return 2.0 * (y - self.y_min) / (self.y_max - self.y_min + self.eps) - 1.0

    def inverse_y(self, y_norm):
        return (
            0.5 * (y_norm + 1.0) * (self.y_max - self.y_min + self.eps)
            + self.y_min
        )


class NPZFieldDataset:
    def __init__(self, path, val_split=0.2, seed=42):
        data = np.load(path, allow_pickle=True)

        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.float32)

        if "boundary" in data:
            self.boundary = data["boundary"].astype(np.float32)
        else:
            self.boundary = np.zeros_like(self.y, dtype=np.float32)

        n = len(self.x)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

        n_val = int(round(n * val_split))

        self.val_idx = idx[:n_val]
        self.train_idx = idx[n_val:]

        self.norm = Normalizer().fit(
            self.x[self.train_idx],
            self.y[self.train_idx],
        )

    def arrays(self, split="train"):
        idx = self.train_idx if split == "train" else self.val_idx

        x = self.norm.transform_x(self.x[idx]).astype(np.float32)
        y = self.norm.transform_y(self.y[idx]).astype(np.float32)
        b = self.boundary[idx].astype(np.float32)

        return x, y, b

    def tf_dataset(self, split="train", batch=2, shuffle=True):
        import tensorflow as tf

        x, y, b = self.arrays(split)

        ds = tf.data.Dataset.from_tensor_slices((x, y, b))

        if shuffle:
            ds = ds.shuffle(
                min(len(x), 2048),
                reshuffle_each_iteration=True,
            )

        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
