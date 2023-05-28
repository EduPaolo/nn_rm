from nn_rm.tensor import Tensor


class SGD:
    def __init__(self, output: Tensor, lr=0.01):
        self.output = output
        self.lr = lr
        self.parameters = self.__get_parameters()


    def step(self):
        breakpoint()
        for p in self.parameters:
            p.arr -= p.grad * self.lr


    def __get_parameters(self):
        parameters = []
        self.__call_parameters(parameters, self.output)
        
        return parameters


    def __call_parameters(self, parameters, output):
        if output.requires_grad:
            parameters.append(output)
        for t in output.children:
            self.__call_parameters(parameters, t)
