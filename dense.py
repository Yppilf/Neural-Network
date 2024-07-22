import numpy as np
from layer import Layer

class Dense(Layer):
    """A fully connected (dense) layer in a neural network.

    Attributes:
    weights (list) - The weight matrix of the layer.
    bias    (list) - The bias vector of the layer.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the Dense layer with random weights and biases.

        Parameters:
        input_size  (int) - The number of input features.
        output_size (int) - The number of output features.
        """
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """Performs the forward pass through the dense layer.

        Parameters:
        input (list) - The input data to the layer.

        Returns:
        (list) - The output of the layer after applying the linear transformation.
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """Performs the backward pass through the dense layer, updating the layer's parameters and computing the gradient for the input.

        Parameters:
        output_gradient (list)  - The gradient of the loss with respect to the layer's output.
        learning_rate   (float) - The learning rate to use for updating the layer's parameters.

        Returns:
        (list) - The gradient of the loss with respect to the layer's input.
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
