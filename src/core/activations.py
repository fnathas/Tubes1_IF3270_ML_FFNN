class Activation:
    def forward(self, x):
        """Menerapkan fungsi aktivasi pada input."""
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, x):
        """Menghitung turunan fungsi aktivasi."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")