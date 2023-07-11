import inspect
from uuid import uuid4


class BaseLayer:
  def __init__(self, name=str(uuid4()), batch_size=None, data_size=None):
    self.name = name
    self.type = type(self).__name__

    self.sized = False
    self.initialized = False

    self.batch_size = batch_size
    self.data_size = data_size

    if self.batch_size is not None and self.data_size is not None:
      self.sized = True

  def get_name(self):
    return "{}({})".format(self.type, self.name)

  def determine_size(self, X):
    shape = X.shape

    if self.batch_size is None or self.batch_size != shape[-1]:
      self.batch_size = shape[-1]

    if self.data_size is None:
      self.data_size = tuple(shape[0:-1])
    else:
      assert self.data_size == tuple(shape[0:-1]), "Data size mismatch in %s(%s)" % (
          self.type, self.name)

    self.sized = True

    if hasattr(self, "__init_params__") and not self.initialized:
      self.__init_params__()

  def forward_prop(self, input):
    if self.__class__.__name__ == "InputLayer":
      X = input
    else:
      self.prevLayer = input
      X = self.prevLayer.A

    if hasattr(self, "__forward_prop__"):
      raise Exception("forward_prop() not implemented for %s(%s)" %
                      (self.type, self.name))

    self.__forward__prop__(X)

  def backward_prop(self, dZ):
    print("backward_prop() not implemented for %s(%s)" %
          (self.type, self.name))

  def backprop(self, dZ):
    print("backprop() not implemented for %s(%s)" % (self.type, self.name))

  def derive_cost(self, Y):
    print("derive_cost() not implemented for %s(%s)" % (self.type, self.name))

  def derive_chain_cost(self, nextLayer):
    print("derive_chain_cost() not implemented for %s(%s)" %
          (self.type, self.name))

  def __update__params__(self, learning_rate):
    print("update() not implemented for %s(%s)" % (self.type, self.name))
