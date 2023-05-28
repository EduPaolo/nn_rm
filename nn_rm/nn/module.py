from abc import ABCMeta, abstractmethod
from nn_rm.tensor import Tensor


class Module(metaclass=ABCMeta):
    def __call__(self, *input) -> Tensor:
        return self.forward(*input)


    @abstractmethod
    def forward(self, *input) -> Tensor:
        pass
