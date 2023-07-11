import numpy as np


def get_predictions(A):
  return np.argmax(A, 0)


def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size


def one_hot(Y, option_count=None):
  max = Y.max() + 1 if option_count is None else option_count

  one_hot_Y = np.zeros((Y.size, max))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T

  return one_hot_Y
