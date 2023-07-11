import numpy as np

from .base import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self, name=None, layer_size=None, batch_size=None, data_size=None, activation=None, activation_deriv=None, cost_function=None):
        super().__init__(name, batch_size, data_size)

        if layer_size is None:
            raise Exception("Layer size must be specified for %s(%s)." % (
                self.type, self.name))

        self.layer_size = layer_size

        if self.batch_size is not None and self.data_size is not None:
            self.__init_params__()

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

        data_size = self.data_size if self.data_size is not None else (1)

        self.W = np.random.rand(self.layer_size, *data_size) - 0.5
        self.b = np.random.rand(self.layer_size, 1) - 0.5

        self.initialized = True

    def __forward__prop__(self, X):
        self.Z = self.W.dot(X) + self.b
        self.A = self.activation(self.Z)

    def __backward__prop__(self, output):
        if self.activation_deriv is None and self.cost_function is None:
            raise Exception("No activation derivative or cost function specified for %s(%s)." % (
                self.type, self.name))

        if self.activation_deriv is not None:
            self.dZ = output.W.T.dot(output.dZ) * self.activation_deriv(self.Z)
        elif self.cost_function is not None:
            self.dZ = self.cost_function(self.A, output)

        # self.dW = 1 / self.batch_size * self.dZ.dot(self.prevLayer.A.T)
        # self.db = 1 / self.batch_size * np.sum(self.dZ)

        self.dW = self.dZ.dot(self.prevLayer.A.T)
        self.db = np.sum(self.dZ)

    def __update__params__(self, alpha):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db
