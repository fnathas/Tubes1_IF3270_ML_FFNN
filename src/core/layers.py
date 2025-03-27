import numpy as np


class Layer:
    def __init__(self):
        """Inisialisasi layer."""
        self.parameters = {}  # weights dan biases
        self.gradients = {}  # gradien weights dan biases
        self.cache = (
            {}
        )  # informasi selama forward pass untuk digunakan selama backward pass

    def forward(self, x):
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, grad_output):
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")

    def update_parameters(self, learning_rate):
        for param_name in self.parameters:
            if param_name in self.gradients:
                self.parameters[param_name] -= (
                    learning_rate * self.gradients[param_name]
                )

    def get_parameters(self):
        return self.parameters

    def get_gradients(self):
        return self.gradients


# (fully connected layer)
class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, activation, weight_initializer=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        if weight_initializer is None:
            self.parameters["weights"] = np.random.randn(
                input_dim, output_dim
            ) * np.sqrt(2.0 / input_dim)
            self.parameters["biases"] = np.zeros((1, output_dim))
        else:
            self.parameters["weights"] = weight_initializer.initialize(
                (input_dim, output_dim)
            )
            self.parameters["biases"] = weight_initializer.initialize((1, output_dim))

        self.gradients["weights"] = np.zeros((input_dim, output_dim))
        self.gradients["biases"] = np.zeros((1, output_dim))

    def forward(self, x):
        self.cache["input"] = x

        z = np.dot(x, self.parameters["weights"]) + self.parameters["biases"]
        self.cache["z"] = z

        output = self.activation.forward(z)

        return output

    def backward(self, grad_output):
        x = self.cache["input"]
        z = self.cache["z"]
        batch_size = x.shape[0]

        dz = grad_output * self.activation.backward(z)

        self.gradients["weights"] = np.dot(x.T, dz) / batch_size

        self.gradients["biases"] = np.sum(dz, axis=0, keepdims=True) / batch_size

        grad_input = np.dot(dz, self.parameters["weights"].T)

        return grad_input
