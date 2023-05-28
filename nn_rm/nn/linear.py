import numpy as np
from nn_rm.tensor import Tensor


class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size

        # initialize weights and bias
        # see pytorch's Linear implementation
        k = np.sqrt(1 / self.input_size)
        self.weights = Tensor(
                np.random.uniform(-k, k, (self.output_size, self.input_size)), True)
        self.bias = Tensor(np.random.uniform(-k, k, (self.output_size)), True)


    def __call__(self, x: Tensor) -> Tensor:
        # parent tensor
        parent_tensor = x @ np.transpose(self.weights) + self.bias

        return parent_tensor
