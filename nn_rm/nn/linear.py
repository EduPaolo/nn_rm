import numpy as np
from nn_rm.tensor import Tensor


class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size


    def __call__(self, x: Tensor) -> Tensor:
        # parent tensor
        breakpoint()
        parent_tensor = x @ np.random.randn(self.input_size, self.output_size).T \
            + np.random.randn(self.output_size)

        return parent_tensor
