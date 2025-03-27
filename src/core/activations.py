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
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)