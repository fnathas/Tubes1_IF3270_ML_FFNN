class Loss:
    def forward(self, y_true, y_pred):
        """Menghitung loss."""
        raise NotImplementedError("Subclasses harus mengimplementasikan forward.")

    def backward(self, y_true, y_pred):
        """Menghitung gradien loss."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")