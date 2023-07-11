import numpy as np

from ..util import one_hot


def CrossEntropy(A, Y):
  n, m = A.shape
  one_hot_Y = one_hot(Y)

  return -np.sum(one_hot_Y * np.log(A + 1e-8)) / m


def CrossEntropy_Derivative(A, Y):
  one_hot_Y = one_hot(Y)

  return A - one_hot_Y
