import numpy as np

from .base import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self, name=None, batch_size=None, data_size=None, filter_size=None, num_filters=None, stride=1, padding=0, activation=None, activation_deriv=None, cost_function=None):
        super().__init__(name, batch_size, data_size)

        if filter_size is None:
            raise Exception("Filter size must be specified for %s(%s)." % (
                self.type, self.name))

        self.filter_size = filter_size
        self.num_filters = num_filters

        self.stride = stride
        self.padding = padding

        if activation is None:
            print("No activation function specified for %s(%s). Defaulting to linear." % (
                self.type, self.name))
            self.activation = lambda x: x

        self.activation = activation
        self.activation_deriv = activation_deriv

        self.cost_function = cost_function

    def __init_params__(self):
        if not self.sized:
            raise Exception(
                "Layer must be sized before initializing parameters.")

        self.W = np.random.rand(self.num_filters, *self.filter_size) - 0.5
        self.b = np.random.rand(self.num_filters, 1) - 0.5

    def __forward__prop__(self, X):
        # Calculate the output dimensions of the convolutional layer
        self.output_height = int(
            (X.shape[0] - self.filter_size[0] + 2 * self.padding) / self.stride) + 1
        self.output_width = int(
            (X.shape[1] - self.filter_size[1] + 2 * self.padding) / self.stride) + 1

        # Initialize the output tensor
        self.Z = np.zeros(
            (self.num_filters, self.output_height, self.output_width))

        # Zero-pad the input if necessary
        if self.padding > 0:
            X = np.pad(X, ((self.padding, self.padding),
                           (self.padding, self.padding)), 'constant')

        # Perform the convolution
        for i in range(self.output_height):
            for j in range(self.output_width):
                self.Z[:, i, j] = np.sum(X[i*self.stride:i*self.stride+self.filter_size[0], j *
                                           self.stride:j*self.stride+self.filter_size[1]] * self.W, axis=(1, 2)) + self.b

        # Apply the activation function
        self.A = self.activation(self.Z)

    def __backward__prop__(self, nextLayer):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Calculate the gradient with respect to the weights and biases
        for i in range(self.output_height):
            for j in range(self.output_width):
                self.dZ[:, i, j] = self.activation_deriv(self.Z[:, i, j])
