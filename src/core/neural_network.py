import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from core.layers import DenseLayer
import core.activations as activations


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation_names,
        loss_function,
        weight_initializer,
    ):
        """Inisialisasi neural network."""
        self.layer_sizes = layer_sizes
        self.activation_names = activation_names
        self.loss_function = loss_function
        self.weight_initializer = weight_initializer

        self.layers = []
        self._initialize_layers()

    def _initialize_layers(self):
        """Inisialisasi layers."""
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            activation_name = self.activation_names[i + 1]

            if activation_name == "linear":
                activation = activations.Linear()
            elif activation_name == "relu":
                activation = activations.ReLU()
            elif activation_name == "sigmoid":
                activation = activations.Sigmoid()
            elif activation_name == "tanh":
                activation = activations.Tanh()
            elif activation_name == "softmax":
                activation = activations.Softmax()
            elif activation_name == "swish":
                activation = activations.Swish()
            elif activation_name == "gelu":
                activation = activations.GELU()
            else:
                raise ValueError(f"Unknown activation function: {activation_name}")

            layer = DenseLayer(
                input_size, output_size, activation, self.weight_initializer
            )
            self.layers.append(layer)

    def forward(self, x):
        """Melakukan forward pass."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        """Melakukan backward pass."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update_parameters(self, learning_rate):
        """Memperbarui parameter model."""
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def display_model(self):
        print("Arsitektur Model:")
        for i, layer in enumerate(self.layers):
            print(
                f"Layer {i + 1}: {layer.input_dim} neurons -> {layer.output_dim} neurons"
            )
            print(f"\nLayer {i+1} Weights:")
            print(layer.parameters["weights"])
            print(f"\nLayer {i+1} Weight Gradients:")
            print(layer.gradients["weights"])
            print(f"\nLayer {i+1} Biases:")
            print(layer.parameters["biases"])
            print(f"\nLayer {i+1} Bias Gradients:")
            print(layer.gradients["biases"])

    def visualize_network(self):
        graph = nx.DiGraph()

        for i, layer_size in enumerate(self.layer_sizes):
            graph.add_node(f"Layer {i+1}", size=layer_size)

        for i in range(len(self.layer_sizes) - 1):
            graph.add_edge(f"Layer {i+1}", f"Layer {i+2}")

        pos = {f"Layer {i+1}": (i, 0) for i in range(len(self.layer_sizes))}

        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=[d[1]["size"] * 100 for d in graph.nodes(data=True)],
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
        )

        for i, layer in enumerate(self.layers):
            weight_mean = np.mean(layer.parameters["weights"])
            gradient_mean = np.mean(layer.gradients["weights"])
            plt.text(
                pos[f"Layer {i+1}"][0] + 0.5,
                pos[f"Layer {i+2}"][1],
                f"W: {weight_mean:.2f}\nG: {gradient_mean:.2f}",
                fontsize=8,
                ha="center",
            )

        plt.title("Neural Network Structure, Weights, and Gradients")
        plt.show()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
