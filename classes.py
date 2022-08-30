import numpy as np


class MLP:
    def __init__(self, layers=[3, 5, 1]):
        self.layers = layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
