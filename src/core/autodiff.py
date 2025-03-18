class Variable:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def backward(self):
        """Menghitung gradien secara otomatis."""
        raise NotImplementedError("Subclasses harus mengimplementasikan backward.")