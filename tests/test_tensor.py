import unittest
import numpy as np
from nn_rm.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_arithmetic_basic(self):
        # basic operations
        arr = [1, 2, 3]
        t = Tensor(arr)
        np.testing.assert_array_equal(np.multiply(t, 2), np.array([2, 4, 6]))
        np.testing.assert_array_equal(np.add(t, 2), np.array([3, 4, 5]))
        np.testing.assert_array_equal(t - 2, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(t > 0, np.array([True, True, True]))
        # check if the class is np.ndarray
        self.assertIsInstance(np.multiply(t, 2), Tensor)
        self.assertIsInstance(np.add(t, 2), Tensor)
        self.assertIsInstance(t - 2, Tensor)
        self.assertIsInstance(t > 0, Tensor)


    def test_arithmetic_advanced(self):
        # matrix multiplication
        arr1 = [[1, 2, 3], [4, 5, 6]]
        arr2 = [[1, 2], [3, 4], [5, 6]]
        t1 = Tensor(arr1)
        t2 = Tensor(arr2)
        np.testing.assert_array_equal(np.matmul(t1, t2), np.array([[22, 28], [49, 64]]))
        np.testing.assert_array_equal(t1 @ t2, np.array([[22, 28], [49, 64]]))
        # check if the class is Tensor
        self.assertIsInstance(np.matmul(t1, t2), Tensor)
        self.assertIsInstance(t1 @ t2, Tensor)
        # nested operations
        x = Tensor([4])
        y = Tensor([5])
        a = 6*x + 3*y
        b = 4*x + 9*y
        z = 2*a + 3*b
        np.testing.assert_array_equal(z, Tensor([261]))


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


    def test_matrix_operations(self):
        # matrix multiplication
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(t1@t2, Tensor([[22, 28], [49, 64]]))

        # matrix addition
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(t1+t2, Tensor([[2, 4, 6], [8, 10, 12]]))

        # matrix subtraction
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(t1-t2, Tensor([[0, 0, 0], [0, 0, 0]]))

        # matrix element-wise multiplication
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(t1*t2, Tensor([[1, 4, 9], [16, 25, 36]]))

    
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
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([[1, 2], [3, 4], [5, 6]])
        z = t1 @ t2

        # run backward
        z.back_prop()
        np.testing.assert_array_equal(t1.grad, Tensor([[1, 3, 5], [2, 4, 6]]))
        np.testing.assert_array_equal(t2.grad, Tensor([[1, 4], [2, 5], [3, 6]]))


if __name__ == '__main__':
    unittest.main()
