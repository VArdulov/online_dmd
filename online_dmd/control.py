import numpy as np
import numpy.linalg as lin

from .online import OnlineDMD

class DMDc():

    def __init__(self, B=None):
        self.A = None
        self.B = B.T if B is not None else B

    def fit(self, X, Y, Z):
        assert (X.shape == Z.shape)
        assert (X.shape[0] == Y.shape[0])
        assert (lin.matrix_rank(Z) >= Z.shape[1])
        if self.B is not None:
            self._known_controller_fit(X, Y, Z)
        else:
            self._unknown_controller_fit(X, Y, Z)


    def _known_controller_fit(self, X, Y, Z):
        U_tilda, Sigma_tilda, V_tilda_inv = lin.svd(X.T, full_matrices=False)
        U_tilda_inv = lin.pinv(U_tilda)
        Sigma_tilda_inv = lin.inv(np.diag(Sigma_tilda))
        V_tilda = lin.pinv(V_tilda_inv)

        diff = Z.T - np.matmul(self.B, Y.T)
        self.A = np.matmul(diff,
                           np.matmul(V_tilda,
                                     np.matmul(Sigma_tilda_inv, U_tilda_inv)))

    def _unknown_controller_fit(self, X, Y, Z):
        p = X.shape[1] # number of features in state
        q = Y.shape[1] # number of features in control

        Omega = np.hstack((X, Y)).T
        if lin.matrix_rank(Omega) < (p + q):
           raise Exception('rank([X, U]) is too small to compute the SVD,'
                           'please collect more samples')

        U_tilda, Sigma_tilda, V_tilda_inv = lin.svd(Omega, full_matrices=False)
        U_tilda_inv = lin.pinv(U_tilda)
        Sigma_tilda_inv = lin.inv(np.diag(Sigma_tilda))
        V_tilda = lin.pinv(V_tilda_inv)


        U_hat, Sigma_hat, V_hat = lin.svd(Z.T)
        U_hat_inv = lin.pinv(U_hat)

        G = np.matmul(U_hat_inv,
                      np.matmul(Z.T,
                                np.matmul(V_tilda,
                                          np.matmul(Sigma_tilda_inv,
                                                    np.matmul(U_tilda_inv, U_hat)))))
        self.A = G[:, :p].T
        self.B = G[:, p:].T
        if self.B.shape[0] != q:
            raise Exception('Something went wrong when computing control matrix!')



class OnlineDMDc(OnlineDMD):

    def __init__(self, B=None, tol=None, track=False):
        super(OnlineDMDc, self).__init__(tol=tol, track=track)
        self.known_controller = not(B is None)
        self.B_k = B
        self.p = 0
        self.q = 0

    def initial_fit(self, X, Y, Z):

        self.p = X.shape[1]  # number of features in state
        self.q = Y.shape[1]  # number of features in control
        Delta = None
        Omega = None

        if self.known_controller:
            Delta = (Z - np.matmul(Y, self.B_k))
            Omega = X
        else:
            Delta = Z
            Omega = np.hstack(X, Y)

        super(OnlineDMDc, self).initial_fit(Omega, Delta)


        if not(self.known_controller):
            self.B_k = self.A_k[self.p:, :]
            self.A_k = self.A_k[:self.p, :]
            if self.B_k.shape[0] != self.q:
                raise Exception('Something went wrong when computing control matrix!')

    def update(self, X, Y, Z):
        if self.iterations == 0:
            self.initial_fit(X, Y, Z)
        elif self.known_controller:
            Delta = (Z - np.matmul(Y, self.B_k))
            super(OnlineDMDc, self).update(X, Delta)
        else:
            for t in range(X.shape[0]):
                if self.track:
                    self.history.append((self.A_k, self.B_k))

                x = X[t, :].reshape(1, -1)
                y = Y[t, :].reshape(1, -1)
                omega = np.hstack((x, y))
                z = Z[t, :].reshape(1, -1)

                self.gamma = (1 / (1 + np.matmul(omega, np.matmul(self.P_k, omega))))[0, 0]
                G_k = np.vstack((self.A_k, self.B_k))
                G_k += self.gamma * (np.matmul(self.P_k.T, np.matmul(omega.T, (z - np.matmul(omega, G_k)))))
                self.P_k -= self.gamma * np.matmul(np.matmul(self.P_k, omega.T), np.matmul(omega, self.P_k))
                self.A_k = G_k[:self.p, :]
                self.B_k = G_k[self.p:, :]
                if self.B.shape[0] != self.q:
                    raise Exception('Something went wrong when computing control matrix!')

                self.iterations += 1
