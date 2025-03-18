class NeuralNetwork:
    def __init__(self):
        """Inisialisasi neural network."""
        raise NotImplementedError("Subclasses harus mengimplementasikan __init__.")

    def forward(self, x):
        """Melakukan forward pass."""
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, grad_output):
        """Melakukan backward pass."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")

    def update_parameters(self, learning_rate):
        """Memperbarui parameter model."""
        raise NotImplementedError("Subclasses harus mengimplementasikan update_parameters.")