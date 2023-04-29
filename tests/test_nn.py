import unittest
import numpy as np
from nn_rm.tensor import Tensor
from nn_rm.nn.module import Module
from nn_rm.nn.linear import Linear
from nn_rm.nn.relu import ReLU


class TestNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        class Net(Module):
            def __init__(self):
                self.fc1 = Linear(3, 2)
                self.relu = ReLU()
                self.fc2 = Linear(2, 3)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                return x

        self.net = Net()


    def test_forward(self):
        x = Tensor(np.random.randn(3))

        y = self.net(x)

        np.testing.assert_array_equal(y, Tensor([0.0, 0.0, 0.0]))
