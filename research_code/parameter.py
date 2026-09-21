import numpy as np


class BilinearParameter():
    def __init__(self, fy, alpha):
        self.fy = np.atleast_1d(fy)
        self.alpha = np.atleast_1d(alpha)


class BoucWenParameter():
    def __init__(self, A, alpha, beta, gamma, n):
        self.A = np.atleast_1d(A)
        self.alpha = np.atleast_1d(alpha)
        self.beta = np.atleast_1d(beta)
        self.gamma = np.atleast_1d(gamma)
        self.n = np.atleast_1d(n)
