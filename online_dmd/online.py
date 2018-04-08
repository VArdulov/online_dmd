import numpy as np
import numpy.linalg as lin


class OnlineDMD:

    def __init__(self, tol=None, track=False):
        self.iterations = 0
        self.tol = tol
        self.P_k = None
        self.A_k = None
        self.gamma = 0
        self.track = track
        self.history = []

    # n - number of state features (columns in X)
    # m - number of observational features (columns in Y)
    # k - number of time steps (snapshots) in X and Y
    def initial_fit(self, X, Y):
        if lin.matrix_rank(X, tol=self.tol) < X.shape[1]:
            raise Exception('Rank of X is less than the dimensionality of X, DMD cannot be performed, '
                            'please change number of features or increase the number of samples. ')
        # X.shape == (k, n)
        # Y.shape == (k, m)

        # lin.pinv(X).shape == (n, k)
        # self.A_k.shape == (n, m)
        self.A_k = np.matmul(lin.pinv(X), Y)
        # self.P_k.shape == (n, n)
        self.P_k = lin.inv(np.matmul(X.T, X))
        self.iterations += 1


    def update(self, X, Y):
        if self.iterations == 0:
            self.initial_fit(X, Y)
        else:
            for t in range(X.shape[0]):
                if self.track:
                    self.history.append(self.A_k)
                x = X[t] #x.shape == (1, n), x.T.shape == (n, 1)
                y = Y[t] #y.shape == (1, m), y.T.shape == (m, 1)
                self.gamma = 1 / (1 + np.matmul(x, np.matmul(self.P_k, x.T)))
                self.A_k += self.gamma * (np.matmul(self.P_k.T, np.matmul(x.T, (y - np.matmul(x, self.A_k)))))
                self.P_k -= self.gamma * np.matmul(np.matmul(self.P_k, x.T), np.matmul(x, self.p_K))
                self.iterations += 1


class WeightedOnlineDMD(OnlineDMD):

    def __init__(self, weight=0.75, tol=None, track=False):
        assert ((weight <= 1) and (weight > 0))
        super(WeightedOnlineDMD, self).__init__(tol=tol, track=track)

        self.sigma = np.sqrt(weight)
        self.rho_inv = (1/weight)


    # n - number of state features (columns in X)
    # m - number of observational features (columns in Y)
    # k - number of time steps (snapshots) in X and Y
    def initial_fit(self, X, Y):

        X_tilda = np.zeros_like(X)
        Y_tilda = np.zeros_like(Y)
        for k in range(X.shape[0]):
            X_tilda[k, :] = X[k, :] * np.power(self.sigma, X.shape[0]-(k+1))
            Y_tilda[k, :] = Y[k, :] * np.power(self.sigma, X.shape[0]-(k+1))

        super(WeightedOnlineDMD, self).initial_fit(X_tilda, Y_tilda)


    def update(self, X, Y):
        if self.iterations == 0:
            self.initial_fit(X, Y)
        else:
            for t in range(X.shape[0]):
                if self.track:
                    self.history.append(self.A_k)
                x = X[t] #x.shape == (1, n), x.T.shape == (n, 1)
                y = Y[t] #y.shape == (1, m), y.T.shape == (m, 1)
                self.gamma = 1 / (1 + np.matmul(x, np.matmul(self.P_k, x.T)))
                self.A_k += self.gamma * (np.matmul(self.P_k.T, np.matmul(x.T, (y - np.matmul(x, self.A_k)))))
                self.P_k -= self.gamma * np.matmul(np.matmul(self.P_k, x.T),np.matmul(x, self.P_k))
                self.P_k *= self.rho_inv
                self.iterations += 1

