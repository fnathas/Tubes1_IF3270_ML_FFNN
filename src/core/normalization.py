import numpy as np


class RMSNorm:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.cache = {}

    def normalize(self, x):
        rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + self.epsilon)
        return x / rms

    def backward(self, x, grad_output):
        rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + self.epsilon)
        x_norm = x / rms

        m = x.shape[1]
        dx_norm = grad_output

        drms = -np.sum(dx_norm * x_norm, axis=1, keepdims=True) / rms

        dx_squared = np.ones_like(x) / (m * rms)

        dx = dx_norm / rms + 2 * x * drms * dx_squared

        return dx
