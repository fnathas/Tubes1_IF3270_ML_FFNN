class Layer:
    def __init__(self):
        """Inisialisasi layer."""
        raise NotImplementedError("Subclasses harus mengimplementasikan __init__.")

    def forward(self, x):
        """Menerapkan operasi layer pada input."""
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, grad_output):
        """Menghitung gradien layer."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")

    def update_parameters(self, learning_rate):
        """Memperbarui parameter layer."""
        raise NotImplementedError("Subclasses harus mengimplementasikan update_parameters.")