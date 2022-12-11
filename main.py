import numpy as np
import scipy.linalg
from scipy.linalg import null_space
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class TightFrame:

    # Enter a Tight Frame as a Matrix with vectors as the columns
    def __init__(self, matrix, transposed=False):
        if transposed:
            self.matrix = matrix
        else:
            self.matrix = matrix.T
        self.ncol, self.nrow = np.shape(matrix)

        # if self.check() is False:
        #     print("WARNING: Given set of vectors is not a Tight Frame!")

    def __str__(self):
        return str(self.matrix)

    def vector_norms(self):
        return np.linalg.norm(self.matrix, axis=0)

    def angles(self, first, second):
        norms = self.vector_norms()
        return np.arccos(np.dot(self.matrix[:, first], self.matrix[:, second]) / (norms[first] * norms[second]))

    def frame_operator(self):
        return self.matrix @ np.transpose(self.matrix)

    def gramian(self):
        return np.transpose(self.matrix) @ self.matrix

    def check(self):
        return np.allclose(self.frame_operator() - np.diag(np.diagonal(self.frame_operator())), 0)

    def constant(self):
        if self.check() is True:
            return np.diagonal(self.frame_operator())[0]
        else:
            return 'It is not a Tight Frame'

    def normalize(self):
        return TightFrame(self.matrix / np.sqrt(self.constant()), transposed=True)

    def complementary(self):
        return TightFrame(np.transpose(null_space(self.matrix)), transposed=True)

    def frame_bounds(self):
        eigen_values = scipy.linalg.eigvalsh(self.frame_operator())
        return np.min(eigen_values), np.max(eigen_values)

    def canonical_dual_frame(self, svd=True):
        if svd:
            U, sigma, V = scipy.linalg.svd(self.matrix)
            return TightFrame(U @ np.c_[np.diag(1/sigma), np.zeros((np.shape(sigma)[0], np.shape(V)[0] - np.shape(sigma)[0]))] @ V, transposed=True)
        else:
            return TightFrame(np.linalg.solve(self.frame_operator(), self.matrix), transposed=True)

    def plot(self, normalize=False):
        assert self.nrow == 2
        if normalize:
            norm_max = np.linalg.norm(self.matrix, 1)
            matrix = self.matrix / norm_max
        else:
            matrix = self.matrix
        V = np.transpose(matrix)

        origin = np.zeros(matrix.shape)  # origin point

        plt.quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=5, label="TF")
        plt.legend()

    def plot_dual(self, normalize=False):

        assert self.nrow == 2

        origin = np.zeros(self.matrix.shape)  # origin point

        # V = np.transpose(self.matrix)
        # plt.quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=5, label="TF")

        W = self.canonical_dual_frame().matrix

        if normalize:
            norm_max = np.linalg.norm(W, 1)
            W = W / norm_max

        W = np.transpose(W)

        plt.quiver(*origin, W[:, 0], W[:, 1], color=['b'], scale=5, label="Dual TF")

        plt.legend()

    def canonical_tight_frame(self, svd=True):
        if svd:
            U, sigma, V = scipy.linalg.svd(self.matrix)
            return TightFrame(U @ np.c_[np.eye(np.shape(U)[0]), np.zeros((np.shape(sigma)[0], np.shape(V)[0] - np.shape(sigma)[0]))] @ V, transposed=True)
        else:
            return TightFrame(scipy.linalg.solve(scipy.linalg.sqrtm(self.frame_operator()), self.matrix), transposed=True)

    def plot_canonical(self, normalize=False):

        assert self.nrow == 2
        origin = np.zeros(self.matrix.shape)  # origin point

        W = self.canonical_tight_frame().matrix
        if normalize:
            norm_max = np.linalg.norm(W, 1)
            W = W / norm_max

        W = np.transpose(W)
        plt.quiver(*origin, W[:, 0], W[:, 1], color=['g'], scale=5, label="Cannonical TF")

        plt.legend()


def entf(n, d):
    """

    :param n: number of vectors
    :param d: dimention od space
    :return: equal-norm tight frame
    """
    if d == 1:
        return np.ones((1, n))
    if d == n:
        return np.eye(n)
    if d < n < 2 * d:
        return np.sqrt(n * d) * np.transpose(scipy.linalg.null_space(entf(n, n - d)))
    if n >= 2 * d > 1:
        return np.concatenate((d * np.eye(d), entf(n - d, d)), axis=1)


def frame_from_gramian(matrix):
    w, sigma, v = scipy.linalg.svd(matrix)
    new_v = scipy.linalg.sqrtm(np.diag(sigma)) @ v
    sliced_v = new_v[~np.all(np.isclose(new_v, 0), axis=1)]
    return TightFrame(sliced_v)


def Welch_inequality(frame):
    oper = frame.frame_operator()
    lhs = np.trace(oper @ oper)
    rhs = (np.trace(oper)) ** 2 / frame.nrow
    return(lhs, rhs)

