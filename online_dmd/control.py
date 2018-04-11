import numpy as np
import numpy.linalg as lin

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



