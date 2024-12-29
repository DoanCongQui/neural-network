import numpy as np

class active_func():
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
        return 1/(1+np.exp(-net))
    
    def _tanh(self, net):
        return (1-np.exp(-2*net))/(1+np.exp(-2*net))

class Perceptron:
    def __init__(self):
        self.active = active_func()

    def tranning(self, X, W, max_epoch):
        pass
        

if __name__ == "__main__":
    print("Pass")

