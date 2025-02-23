"""
@ MAIN all algorithm neural networking
@ dq
@ Github: github.com/DoanCongQui
"""

from lib import NeuralNertwork

import numpy as np

if __name__ == "__main__":
    # Ex2a:
    # X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    # W = np.array([0.1, 0.3, 0.5])
    # d = np.array([0, 0, 0, 1])

    # Ex2b:
    # X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    # W = np.array([0.1, 0.3, 0.5])
    # d = np.array([0, 1, 1, 1])

    # Ex3:
    X = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0.958, 1.043, 1.907, 0.780, 0.579, 0.003, 0.001, 0.014], [0.003, 0.001, 0.003, 0.002, 0.001, 0.105, 1.748, 1.839]])
    W = np.array([0.1, 0.3, 0.5])
    d = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    nn = NeuralNertwork(X, W, d, 0.1, 1000)
    nn.Perceptron()
