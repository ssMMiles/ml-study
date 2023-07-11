from lib.activations import ReLU, ReLU_Derivative, Softmax
from lib.cost import CrossEntropy, CrossEntropy_Derivative
from lib.layers import InputLayer, DenseLayer
from lib.networks.sequential import SequentialModel
from lib.util import get_predictions, get_accuracy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

test_data = pd.read_csv('../datasets/mnist/test.csv')
train_data = pd.read_csv('../datasets/mnist/train.csv')

Y_test = test_data['label'].values
X_test = test_data.drop('label', axis=1).values.T / 255.

Y_train = train_data['label'].values
X_train = train_data.drop('label', axis=1).values.T / 255.

alpha = 0.01
iterations = 250
batch_size = None

schema = [
    InputLayer("input"),
    DenseLayer("hidden", 784, 100, None, ReLU, ReLU_Derivative),
    DenseLayer("output", 10, 100, None, Softmax, None, CrossEntropy_Derivative)
]

model = SequentialModel(schema)

print("Model:")
model.display()

print("Training...")
model.run_gradient_descent(X_train, Y_train, alpha, iterations, batch_size)

print("Verifying against test data...")
A = model.forward_prop(X_test)
predictions = get_predictions(A)

print("Achieved %f%% accuracy against the test set. Iterations: %d Alpha: %f Cross Entropy: %f" %
      ((get_accuracy(predictions, Y_test) * 100),
       iterations, alpha, CrossEntropy(A, Y_test))
      + ((" Batch Size: %d" % batch_size) if batch_size != None else " (No batch size)"))

model.save("mnist.model")

print("Finished training and saved the output model to mnist.model")
