import pickle

from rich.table import Table, Column
from rich.live import Live
from rich.console import Console

from pathlib import Path
from uuid import uuid4

from ..util import get_predictions, get_accuracy
from ..cost import CrossEntropy


class SequentialModel(object):
    def __init__(self, layers=[], name=str(uuid4())):
        self.layers = layers
        self.name = name

        self.training = False

        self.console = Console()
        self.training_console = Console()

    def forward_prop(self, X):
        for i in range(len(self.layers)):
            layer = self.layers[i]

            if i == 0:
                prevLayer = X
                input = X
            else:
                prevLayer = self.layers[i - 1]
                input = prevLayer.A

            batch_size_changed = layer.sized and layer.batch_size != X.shape[-1]

            if layer.sized is False or batch_size_changed is True:
                layer.determine_size(input)

                if hasattr(self, 'live'):
                    self.live.update(self.generate_model_table(), refresh=True)

            layer.forward_prop(prevLayer)

            if hasattr(self, 'live'):
                self.live.update(self.generate_model_table(), refresh=True)

        return self.layers[-1].A

    def backward_prop(self, Y):
        for i in range(len(self.layers) - 1, 0, -1):
            isLastLayer = i == len(self.layers) - 1
            layer = self.layers[i]

            output = Y if isLastLayer else self.layers[i + 1]
            layer.__backward__prop__(output)

    def predict(self, X):
        A = self.forward_prop(X)
        predictions = get_predictions(A)

        return predictions

    def __update__params__(self, alpha):
        for i in range(1, len(self.layers)):
            self.layers[i].__update__params__(alpha)

    def run_gradient_descent(self, X, Y, alpha, iterations, batch_size=None):
        if self.training:
            self.console.print("Model is already training!")
            return

        self.training = True

        self.iterations = iterations
        self.batch_size = batch_size
        self.alpha = alpha

        for i in range(self.iterations):
            self.current_iteration = i + 1

            if self.batch_size == None:
                A = self.forward_prop(X)

                predictions = get_predictions(A)

                self.accuracy = get_accuracy(predictions, Y) * 100
                self.cross_entropy = CrossEntropy(A, Y)

                self.backward_prop(Y)

                self.__update__params__(self.alpha)
            else:
                # Divide the training data into mini-batches
                mini_batches = [(X.T[i:i+self.batch_size].T, Y[i:i+self.batch_size])
                                for i in range(0, len(X), self.batch_size)]

                accuracies = []
                cross_entropies = []

                # Loop over the mini-batches
                for X_batch, Y_batch in mini_batches:
                    # Perform forward and backpropagation on the mini-batch
                    A = self.forward_prop(X_batch)

                    predictions = get_predictions(A)
                    accuracies.append(get_accuracy(predictions, Y_batch) * 100)

                    cross_entropies.append(CrossEntropy(A, Y_batch))

                    self.backward_prop(Y_batch)

                    # Update the weights and biases using the gradients
                    self.__update__params__(self.alpha)

                self.accuracy = sum(accuracies) / len(accuracies)
                self.cross_entropy = sum(
                    cross_entropies) / len(cross_entropies)

            if hasattr(self, 'live_training'):
                self.live_training.update(
                    self.generate_training_table(), refresh=True)

        self.training = False
        return self

    def test_accuracy(self, X, Y):
        predictions = self.predict(X)
        accuracy = get_accuracy(predictions, Y) * 100

        return accuracy

    def generate_model_table(self):
        model = Table(
            Column("Layer", justify="right"),
            Column("Type", justify="right"),
            Column("Data Shape", justify="right"),
            Column("Layer Size", justify="right"),
            title="Sequential Model \"{}\"".format(self.name)
        )

        for layer in self.layers:
            layer_size = "None"
            if hasattr(layer, 'layer_size'):
                layer_size = str(layer.layer_size)

            shape = (layer.data_size or ("N/A",))

            if len(shape) == 1:
                shape = shape[0]

            model.add_row(layer.name, layer.type, str(shape), layer_size)

        return model

    def generate_training_table(self):
        training = Table(
            Column("Iteration", justify="right"),
            Column("Batch Size", justify="right"),
            Column("Learning Rate", justify="right"),
            Column("Accuracy", justify="right"),
            Column("Cross Entropy", justify="right"),
            title="Training..."
        )

        batch_size = self.batch_size or "Full ({})".format(
            self.layers[0].batch_size if hasattr(
                self.layers[0], 'batch_size') else "N/A"
        )

        training.add_row("{}/{}".format(self.current_iteration, self.iterations), str(batch_size),
                         str(self.alpha), "%.5f" % self.accuracy, "%.5f" % self.cross_entropy)

        return training

    def display(self):
        if not hasattr(self, 'live'):
            self.live = Live("Initializing...",
                             refresh_per_second=4, console=self.console)
            self.live_training = Live("The model is not currently being trained.",
                                      refresh_per_second=4, console=self.training_console)

        if self.training:
            self.live.update(self.generate_model_table(), refresh=True)
            self.live_training.update(
                self.generate_training_table())
        else:
            self.live.update(self.generate_model_table(), refresh=True)

    def save(self):
        data = open('{}.model'.format(self.name), 'wb')
        pickle.dump(self.layers, data, -1)

        data.close()

    def load(path, name=None):
        data = open(path, 'rb')
        layers = pickle.load(data)

        data.close()

        name = name or Path(path).stem

        return SequentialModel(layers, name)
