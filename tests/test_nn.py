import unittest
import numpy as np
from nn_rm.tensor import Tensor
from nn_rm.nn.module import Module
from nn_rm.nn.linear import Linear
from nn_rm.nn.relu import ReLU
from nn_rm.optim.SGD import SGD


class TestNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        class Net(Module):
            def __init__(self):
                self.fc1 = Linear(3, 2)
                self.relu = ReLU()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                return x

        self.net = Net()


    def test_forward(self):
        x = Tensor(np.random.randn(3))

        y = self.net(x)

        # test random seed
        np.testing.assert_array_almost_equal(x, Tensor([0.49671414, -0.1382643, 0.64768857]))

        # test forward
        np.testing.assert_array_almost_equal(y, Tensor([0, 0.892099]))


    def test_backward(self):
        x = Tensor(np.random.randn(3))

        y = self.net(x)

        # optimize
        optimizer = SGD(y, lr=0.01)

        # backprop
        y.back_prop()
        # step
        optimizer.step()

        # test backward
        np.testing.assert_array_almost_equal(x.grad, Tensor([0, 0, 0]))
