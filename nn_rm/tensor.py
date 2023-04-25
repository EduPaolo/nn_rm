from __future__ import annotations
from typing import Callable, Any

import numpy as np
import numpy.typing as npt



class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, arr: npt.ArrayLike):
        self.arr = np.array(arr, dtype=np.float32)
        self.grad: Any = 0
        self.backward: Callable = lambda: None
        self.children: list = []


    def back_prop(self):
        self.grad = 1
        self.__wrap_backward()


    def __wrap_backward(self):
        self.backward()
        for child in self.children:
            child.__wrap_backward()


    def __repr__(self) -> str:
        return f"Tensor({self.arr})"


    def __array__(self) -> np.ndarray:
        return self.arr


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        tensor_inputs = [isinstance(i, Tensor) for i in inputs]
        if any(tensor_inputs):
            new_inputs = []
            for i in inputs:
                if isinstance(i, Tensor):
                    new_inputs.append(i.arr)
                else:
                    new_inputs.append(i)
            result = ufunc(*new_inputs, **kwargs)
            return Tensor(result)
        else:
            return NotImplemented


    def __add__(self, other):
        # to tensor
        other = other if isinstance(other, Tensor) else Tensor(other)

        # numpy addition
        parent_tensor = super().__add__(other)

        # add children
        parent_tensor.children.append(self)
        parent_tensor.children.append(other)

        def backward():
            self.grad = self.grad + parent_tensor.grad*1
            other.grad = other.grad + parent_tensor.grad*1

        parent_tensor.backward = backward

        return parent_tensor


    def __sub__(self, other):
        # to tensor
        other = other if isinstance(other, Tensor) else Tensor(other)

        # numpy subtraction
        parent_tensor = super().__sub__(other)

        # add children
        parent_tensor.children.append(self)
        parent_tensor.children.append(other)

        def backward():
            self.grad = self.grad + parent_tensor.grad*1
            other.grad = other.grad + parent_tensor.grad*-1

        parent_tensor.backward = backward

        return parent_tensor


    def __mul__(self, other):
        # to tensor
        other = other if isinstance(other, Tensor) else Tensor(other)

        # numpy multiplication
        parent_tensor = super().__mul__(other)

        # add children
        parent_tensor.children.append(self)
        parent_tensor.children.append(other)

        def backward():
            self.grad = self.grad + parent_tensor.grad*other
            other.grad = other.grad + parent_tensor.grad*self

        parent_tensor.backward = backward

        return parent_tensor


    def __matmul__(self, other):
        # to tensor
        other = other if isinstance(other, Tensor) else Tensor(other)

        # numpy matmul
        parent_tensor = super().__matmul__(other)

        # add children
        parent_tensor.children.append(self)
        parent_tensor.children.append(other)

        def backward():
            self.grad += parent_tensor.grad*np.transpose(other)
            other.grad += parent_tensor.grad*np.transpose(self)

        parent_tensor.backward = backward

        return parent_tensor


    def __radd__(self, other):
        return self + other


    def __rmul__(self, other):
        return self * other


    def __rsub__(self, other):
        return self - other


    def __rmatmul__(self, other):
        return self @ other
