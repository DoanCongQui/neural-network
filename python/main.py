"""
@ MAIN all algorithm neural networking
@ dq
@ Github: github.com/DoanCongQui
"""

from lib import NeuralNertwork

import numpy as np

if __name__ == "__main__":
    X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    W = np.array([0.1, 0.3, 0.5])
    d = np.array([0, 0, 0, 1])

    nn = NeuralNertwork(X, W, d, 0.1, 1000)
    nn.Perceptron()
