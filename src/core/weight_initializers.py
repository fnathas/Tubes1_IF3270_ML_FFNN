import numpy as np

class WeightInitializer:
    def initialize(self, shape):
        raise NotImplementedError("Subclasses harus mengimplementasikan initialize.")

# Zero Initialization
class ZeroInitializer(WeightInitializer):
    def initialize(self, shape):
        return np.zeros(shape)

# Random Uniform Initialization
class UniformInitializer(WeightInitializer):
    def __init__(self, lower_bound=-0.1, upper_bound=0.1, seed=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.lower_bound, self.upper_bound, shape)

# Random Normal Initialization
class NormalInitializer(WeightInitializer):
    def __init__(self, mean=0, variance=0.01, seed=None):
        self.mean = mean
        self.variance = variance
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, np.sqrt(self.variance), shape)

# Bonus
# Xavier Initialization
class XavierInitializer(WeightInitializer):
    def initialize(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

# He Initialization
class HeInitializer(WeightInitializer):
    def initialize(self, shape):
        fan_in, _ = shape
        return np.random.normal(0, np.sqrt(2 / fan_in), shape)