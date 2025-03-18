class Metric:
    def calculate(self, y_true, y_pred):
        """Menghitung metrik evaluasi."""
        raise NotImplementedError("Subclasses harus mengimplementasikan calculate.")