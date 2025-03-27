import numpy as np


class Activation:
    def forward(self, x):
        """Menerapkan fungsi aktivasi pada input."""
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, x):
        """Menghitung turunan fungsi aktivasi."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")


class Linear(Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        return np.ones_like(x)


class Swish(Activation):
    def forward(self, x):
        return x * (1 / (1 + np.exp(-x)))

    def backward(self, x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)


class GELU(Activation):
    def forward(self, x):
        return (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )

    def backward(self, x):
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf
