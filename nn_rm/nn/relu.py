import numpy as np
from nn_rm.tensor import Tensor


class ReLU:
    def __init__(self):
        pass


    def __call__(self, x: Tensor) -> Tensor:
        # parent tensor
        parent_tensor = np.maximum(x, 0)

        # add children
        parent_tensor.children.append(x)

        def backward():
            x.grad = x.grad + parent_tensor.grad * (x > 0)

        parent_tensor.backward = backward

        return parent_tensor
