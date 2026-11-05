"""NumPy-only inference for an exported sensor-input device actor."""

from pathlib import Path

import numpy as np


class DeviceController:
    """Map raw SI input [..., input_dim] to ordered normalized device commands."""

    def __init__(self, weights: str | Path) -> None:
        with np.load(weights, allow_pickle=False) as arrays:
            self.mean, self.var = arrays["mean"], arrays["var"]
            self.layers = [(arrays[f"kernel_{i}"], arrays[f"bias_{i}"]) for i in range(int(arrays["layers"]))]

    def __call__(self, sensors: np.ndarray) -> np.ndarray:
        values = np.asarray(sensors, dtype=np.float32)
        if values.ndim == 0 or values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite raw sensor input [..., {len(self.mean)}]")
        values = (values - self.mean) / np.sqrt(self.var + 1e-8)
        for index, (kernel, bias) in enumerate(self.layers):
            values = values @ kernel + bias
            if index < len(self.layers) - 1:
                values = np.tanh(values)
        return np.clip(values, -1.0, 1.0)
