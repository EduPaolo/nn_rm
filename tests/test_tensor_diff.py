import unittest
import numpy as np
from nn_rm.tensor import Tensor
from nn_rm.nn.relu import ReLU


class TestTensorDiff(unittest.TestCase):
    def setUp(self):
        self.relu = ReLU()


    def test_diff_rm(self):
        # build graph
        x = Tensor([4])
        y = Tensor([5])
        a = 6*x + 3*y
        b = 4*x + 9*y
        z = 2*a + 3*b

        # run backward
        z.back_prop()
        np.testing.assert_array_equal(x.grad, Tensor(24))
        np.testing.assert_array_equal(y.grad, Tensor(33))


    def test_matrix_diff_basic(self):
        # build graph
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = Tensor([[1, 2, 3], [4, 5, 6]])
        a = 6*x + 3*y
        b = 4*x + 9*y
        z = 2*a + 3*b

        # run backward
        z.back_prop()
        np.testing.assert_array_equal(x.grad, Tensor([[24, 24, 24], [24, 24, 24]]))
        np.testing.assert_array_equal(y.grad, Tensor([[33, 33, 33], [33, 33, 33]]))


    def test_matrix_diff_advanced(self):
        # build graph
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        z = t1 @ t2

        # run backward
        z.back_prop()
        np.testing.assert_array_equal(t1.grad, Tensor([[3, 7, 11], [3, 7, 11]]))
        np.testing.assert_array_equal(t2.grad, Tensor([[5, 5], [7, 7], [9, 9]]))


    def test_relu_diff(self):
        # build graph
        x = Tensor([1, -1, 0])
        y = self.relu(x)

        # run backward
        y.back_prop()
        np.testing.assert_array_equal(x.grad, Tensor([1, 0, 0]))
