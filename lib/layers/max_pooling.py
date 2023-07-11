from .base import BaseLayer


class MaxPoolingLayer(BaseLayer):
  def __forward__prop__(self, X):
    self.A = X.transpose(2, 0, 1).reshape(self.batch_size, -1).T

  def __backward__prop__(self, nextLayer):
    self.dZ = nextLayer.dZ

  def __update__params__(self, learning_rate):
    pass
