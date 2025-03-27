import numpy as np


class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, y_true, y_pred):
        raise NotImplementedError("Backward method not implemented.")


class MeanSquaredError(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        return 2 * (y_pred - y_true) / n_samples


class BinaryCrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)


class CategoricalCrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def backward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]
