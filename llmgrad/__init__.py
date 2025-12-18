import ctensor

Tensor = ctensor.Tensor
embedding_concat = ctensor.embedding_concat
embedding_forward = ctensor.embedding_forward

from .nn import Module, Linear, Embedding, Tanh, Sequential
from .optim import SGD
