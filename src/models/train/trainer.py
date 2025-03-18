class Trainer:
    def __init__(self, model, optimizer, loss):
        """Inisialisasi trainer."""
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, x_batch, y_batch):
        """Melakukan satu langkah pelatihan."""
        raise NotImplementedError("Subclasses harus mengimplementasikan train_step.")

    def evaluate(self, x_val, y_val):
        """Mengevaluasi model pada data validasi."""
        raise NotImplementedError("Subclasses harus mengimplementasikan evaluate.")