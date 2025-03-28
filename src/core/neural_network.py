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

    def plot_weight_distribution(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = range(len(self.layers))

        num_layers = len(layers_to_plot)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

        if num_layers == 1:
            axes = [axes]

        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue

            layer = self.layers[layer_idx]
            if not hasattr(layer, "parameters") or "weights" not in layer.parameters:
                print(f"Warning: Layer {layer_idx} has no weights. Skipping.")
                continue

            weights = layer.parameters["weights"].flatten()
            axes[i].hist(weights, bins=50, alpha=0.7, color="blue")
            axes[i].set_title(f"Layer {layer_idx+1} Weight Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = range(len(self.layers))

        num_layers = len(layers_to_plot)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

        if num_layers == 1:
            axes = [axes]

        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue

            layer = self.layers[layer_idx]
            if not hasattr(layer, "gradients") or "weights" not in layer.gradients:
                print(f"Warning: Layer {layer_idx} has no weight gradients. Skipping.")
                continue

            gradients = layer.gradients["weights"].flatten()
            axes[i].hist(gradients, bins=50, alpha=0.7, color="red")
            axes[i].set_title(f"Layer {layer_idx+1} Gradient Distribution")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        batch_size=32,
        learning_rate=0.01,
        epochs=100,
        l1_lambda=0,
        l2_lambda=0,
        momentum=0,
        use_rmsnorm=False,
        verbose=1,
        early_stopping_patience=None,
    ):
        from tqdm import tqdm

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        if momentum > 0:
            velocity = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "parameters") and "weights" in layer.parameters:
                    velocity[i] = {
                        "weights": np.zeros_like(layer.parameters["weights"]),
                        "biases": np.zeros_like(layer.parameters["biases"]),
                    }

        if use_rmsnorm:
            rmsnorm_cache = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "parameters") and "weights" in layer.parameters:
                    rmsnorm_cache[i] = {
                        "weights_rms": np.zeros_like(layer.parameters["weights"]),
                        "biases_rms": np.zeros_like(layer.parameters["biases"]),
                    }

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            train_losses = []

            batch_iterator = range(0, len(X_train), batch_size)
            if verbose == 1:
                batch_iterator = tqdm(batch_iterator, desc=f"Epoch {epoch+1}/{epochs}")

            for i in batch_iterator:
                X_batch = X_train_shuffled[i : i + batch_size]
                y_batch = y_train_shuffled[i : i + batch_size]

                y_pred = self.forward(X_batch)

                loss = self.loss_function.forward(y_batch, y_pred)

                reg_loss = 0

                if l1_lambda > 0 or l2_lambda > 0:
                    for layer in self.layers:
                        if (
                            hasattr(layer, "parameters")
                            and "weights" in layer.parameters
                        ):
                            # L1
                            if l1_lambda > 0:
                                reg_loss += l1_lambda * np.sum(
                                    np.abs(layer.parameters["weights"])
                                )

                            # L2
                            if l2_lambda > 0:
                                reg_loss += (l2_lambda / 2) * np.sum(
                                    np.square(layer.parameters["weights"])
                                )

                total_loss = loss + reg_loss
                train_losses.append(total_loss)

                grad = self.loss_function.backward(y_batch, y_pred)
                self.backward(grad)

                for layer in self.layers:
                    if hasattr(layer, "parameters") and "weights" in layer.parameters:
                        if "weights" in layer.gradients:
                            if l1_lambda > 0:
                                l1_grad = l1_lambda * np.sign(
                                    layer.parameters["weights"]
                                )
                                layer.gradients["weights"] += l1_grad

                            if l2_lambda > 0:
                                l2_grad = l2_lambda * layer.parameters["weights"]
                                layer.gradients["weights"] += l2_grad

                for j, layer in enumerate(self.layers):
                    if hasattr(layer, "parameters") and "weights" in layer.parameters:
                        if "weights" in layer.gradients:
                            if momentum > 0 and j in velocity:
                                velocity[j]["weights"] = (
                                    momentum * velocity[j]["weights"]
                                    - learning_rate * layer.gradients["weights"]
                                )
                                velocity[j]["biases"] = (
                                    momentum * velocity[j]["biases"]
                                    - learning_rate * layer.gradients["biases"]
                                )

                                layer.parameters["weights"] += velocity[j]["weights"]
                                layer.parameters["biases"] += velocity[j]["biases"]
                            elif use_rmsnorm and j in rmsnorm_cache:
                                epsilon = 1e-8
                                decay_rate = 0.999

                                rmsnorm_cache[j][
                                    "weights_rms"
                                ] = decay_rate * rmsnorm_cache[j]["weights_rms"] + (
                                    1 - decay_rate
                                ) * np.square(
                                    layer.gradients["weights"]
                                )

                                rmsnorm_cache[j][
                                    "biases_rms"
                                ] = decay_rate * rmsnorm_cache[j]["biases_rms"] + (
                                    1 - decay_rate
                                ) * np.square(
                                    layer.gradients["biases"]
                                )

                                weights_update = (
                                    learning_rate
                                    * layer.gradients["weights"]
                                    / (
                                        np.sqrt(rmsnorm_cache[j]["weights_rms"])
                                        + epsilon
                                    )
                                )
                                biases_update = (
                                    learning_rate
                                    * layer.gradients["biases"]
                                    / (
                                        np.sqrt(rmsnorm_cache[j]["biases_rms"])
                                        + epsilon
                                    )
                                )

                                layer.parameters["weights"] -= weights_update
                                layer.parameters["biases"] -= biases_update
                            else:
                                # Standard gradient descent
                                layer.parameters["weights"] -= (
                                    learning_rate * layer.gradients["weights"]
                                )
                                layer.parameters["biases"] -= (
                                    learning_rate * layer.gradients["biases"]
                                )

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.loss_function.forward(y_val, val_pred)

                val_accuracy = None
                if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                    val_pred_classes = np.argmax(val_pred, axis=1)
                    val_true_classes = np.argmax(y_val, axis=1)
                    val_accuracy = np.mean(val_pred_classes == val_true_classes)

                history["val_loss"].append(val_loss)
                if val_accuracy is not None:
                    history["val_accuracy"].append(val_accuracy)

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0

                        best_weights = {}
                        for j, layer in enumerate(self.layers):
                            if (
                                hasattr(layer, "parameters")
                                and "weights" in layer.parameters
                            ):
                                best_weights[j] = {
                                    param_name: np.copy(param_value)
                                    for param_name, param_value in layer.parameters.items()
                                }
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose > 0:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                        break

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            if verbose > 0:
                status = f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}"
                if X_val is not None and y_val is not None:
                    status += f", Val Loss: {val_loss:.4f}"
                    if val_accuracy is not None:
                        status += f", Val Acc: {val_accuracy:.4f}"
                print(status)

        if early_stopping_patience is not None and best_weights:
            for j, layer in enumerate(self.layers):
                if (
                    j in best_weights
                    and hasattr(layer, "parameters")
                    and "weights" in layer.parameters
                ):
                    for param_name, param_value in best_weights[j].items():
                        layer.parameters[param_name] = param_value

        return history

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
