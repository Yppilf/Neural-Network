import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    """A convolutional layer in a neural network.

    Attributes:
    input_shape     (tuple) - The shape of the input (depth, height, width).
    kernel_size     (int)   - The size of the convolutional kernels (filters).
    depth           (int)   - The number of convolutional kernels (filters).
    output_shape    (tuple) - The shape of the output (depth, height, width).
    kernels_shape   (tuple) - The shape of the kernels (depth, input_depth, height, width).
    kernels         (list)  - The convolutional kernels (filters).
    biases          (list)  - The biases for the convolutional layer.
    """

    def __init__(self, input_shape, kernel_size, depth):
        """Initializes the Convolutional layer with random kernels and biases.

        Parameters:
        input_shape (tuple) - The shape of the input (depth, height, width).
        kernel_size (int)   - The size of the convolutional kernels (filters).
        depth       (int)   - The number of convolutional kernels (filters).
        """
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        """Performs the forward pass through the convolutional layer.

        Parameters:
        input (list) - The input data to the layer.

        Returns:
        (list) - The output of the layer after applying the convolution operation.
        """
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Performs the backward pass through the convolutional layer, updating the layer's parameters and computing the gradient for the input.

        Parameters:
        output_gradient (list)  - The gradient of the loss with respect to the layer's output.
        learning_rate   (float) - The learning rate to use for updating the layer's parameters.

        Returns:
        (list) - The gradient of the loss with respect to the layer's input.
        """
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
