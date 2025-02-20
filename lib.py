import numpy as np


class active_func:
    def __init__(self) -> None:
        pass

    def _step(self, net):
        if net >= 0:
            return 1
        else:
            return 0

    def _sign(self, net):
        if net >= 0:
            return 1
        else:
            return -1

    def _linear(self, net):
        return net

    def _ReLU(self, net):
        if net >= 0:
            return net
        else:
            return 0

    def _sigmoid(self, net):
        return 1 / (1 + np.exp(-net))

    def _tanh(self, net):
        return (1 - np.exp(-2 * net)) / (1 + np.exp(-2 * net))


class NeuralNertwork:
    def __init__(self, X, W, d, teta, max_epoch):
        self.active = active_func()
        self.X = X
        self.W = W
        self.d = d
        self.teta = teta
        self.max_epoch = max_epoch

    def Perceptron(self):
        epoch = 0
        size = len(self.X) + 1
        print(size)
        E = 0
        while epoch < self.max_epoch:
            for i in range(size):
                net = self.W.T @ self.X[:, i]
                print(net)
                y = self.active._step(net)
                self.W += self.teta * (self.d[i] - y) * self.X[:, i]
                E = E + 0.5 * (self.d[i] - y) ** 2
                print(self.W)
                print(E)

            epoch += 1
            if E == 0:
                break


if __name__ == "__main__":
    X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    W = np.array([0.1, 0.3, 0.5])
    d = np.array([0, 0, 0, 1])

    nn = NeuralNertwork(X, W, d, 0.1, 1)
    nn.Perceptron()
