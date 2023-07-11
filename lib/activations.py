# Activation Functions + Derivatives

import numpy as np


def ReLU(Z):
  return np.maximum(Z, 0)


def ReLU_Derivative(Z):
  return Z > 0


def Softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A


def Sigmoid(Z):
  A = 1 / 1 + np.exp(-Z)
  return A
