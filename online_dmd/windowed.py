import numpy as np
import numpy.linalg as lin

from .online import OnlineDMD

class WindowedDMD(OnlineDMD):

    def __init__(self, window=10, tol=None, track=False):
        assert window > 0

        super(WindowedDMD, self).__init__(tol=tol, track=track)
        self.w = window
        self.X_window = None
        self.Y_window = None
        self.C_inv = lin.inv(np.diag([-1, 1]))

    # n - number of state features (columns in X)
    # m - number of observational features (columns in Y)
    # k - number of time steps (snapshots) in X and Y
    def initial_fit(self, X, Y):
        if X.shape[0] < self.w:
            raise Exception('Not enough samples provided to initilize the window')
        if lin.matrix_rank(X, tol=self.tol) < X.shape[1]:
            raise Exception('Rank of X is less than the dimensionality of X, DMD cannot be performed, '
                            'please change number of features or increase the number of samples. ')


        super(WindowedDMD, self).initial_fit(X[0:self.w], Y[0:self.w])
        self.X_window = X[0:self.w]
        self.Y_window = Y[0:self.w]
        self.update(X[self.w:], Y[self.w:])


    def update(self, X, Y):
        if self.iterations == 0:
            self.initial_fit(X, Y)
        else:
            for t in range(X.shape[0]):
                if self.track:
                    self.history.append(self.A_k)

                U = np.vstack((self.X_window[0, :], X[t]))
                V = np.vstack((self.Y_window[0, :], Y[t]))

                self.gamma = lin.inv((self.C_inv + np.matmul(U, np.matmul(self.P_k, U.T))))
                self.A_k += np.matmul((V - np.matmul(U, self.A_k)).T, np.matmul(self.gamma, np.matmul(U, self.P_k))).T
                self.P_k -= np.matmul(np.matmul(self.P_k, np.matmul(U.T, self.gamma)), np.matmul(U, self.P_k))
                self.X_window = np.vstack((self.X_window[1:, :], X[t]))
                self.Y_window = np.vstack((self.Y_window[1:, :], Y[t]))
                self.iterations += 1

class WeightedWindowedDMD(WindowedDMD):

    def __init__(self, window=10, weight=0.75, tol=None, track=False):
        assert window > 0
        assert ((weight <= 1) and (weight > 0))

        super(WeightedWindowedDMD, self).__init__(window=window, tol=tol, track=track)
        self.sigma = np.sqrt(weight)
        self.C_inv = lin.inv(np.diag([np.power(-weight, window-1), 1]))
        self.rho_inv = (1/weight)

    def initial_fit(self, X, Y):
        X_tilda = np.zeros_like(X[0:self.w, :])
        Y_tilda = np.zeros_like(Y[0:self.w, :])
        for k in range(X.shape[0]):
            X_tilda[k, :] = X[k, :] * np.power(self.sigma, X.shape[0] - (k + 1))
            Y_tilda[k, :] = Y[k, :] * np.power(self.sigma, X.shape[0] - (k + 1))

        super(WeightedWindowedDMD, self).initial_fit(X_tilda, Y_tilda)
        self.update(X[self.w:, :], Y[self.w:, :])

    def update(self, X, Y):
        if self.iterations == 0:
            self.initial_fit(X, Y)
        else:
            for t in range(X.shape[0]):
                if self.track:
                    self.history.append(self.A_k)
                U = np.vstack((self.X_window[0, :], X[t]))
                V = np.vstack((self.Y_window[0, :], Y[t]))

                self.gamma = lin.inv((self.C_inv + np.matmul(U, np.matmul(self.P_k, U.T))))
                self.A_k += np.matmul((V - np.matmul(U, self.A_k)).T, np.matmul(self.gamma, np.matmul(U, self.P_k))).T
                self.P_k -= np.matmul(np.matmul(self.P_k, np.matmul(U.T, self.gamma)), np.matmul(U, self.P_k))
                self.P_k *= self.rho_inv
                self.X_window = np.vstack((self.X_window[1:, :], X[t]))
                self.Y_window = np.vstack((self.Y_window[1:, :], Y[t]))
                self.iterations += 1
