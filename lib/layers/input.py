from .base import BaseLayer


class InputLayer(BaseLayer):
    def __forward__prop__(self, X):
      self.A = X
