import numpy as np
from scipy.linalg import null_space

class TightFrame:

    # Enter a Tight Frame as a Matrix with vectors as the columns
    def __init__(self, matrix):
        self.matrix = matrix
        self.nrow, self.ncol = np.shape(matrix)

        if self.check() is False:
            print("WARNING: Given set of vectors is not a Tight Frame!")

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
        return np.allclose(self.frame_operator() - np.diag(np.diagonal(self.frame_operator())),0)

    def constant(self):
        if self.check() is True:
            return np.diagonal(self.frame_operator())[0]
        else:
            return 'It is not a Tight Frame'

    def normalize(self):
        return TightFrame(self.matrix / np.sqrt(self.constant()))

    def complementary(self):
        return TightFrame(np.transpose(null_space(self.matrix)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    x = TightFrame(np.array([[1,0,-0.5],[0,1,-0.5]]))
    print(x)
    print(x.vector_norms())
    print(x.angles(0,1))
    print(x.check())

    y = TightFrame(np.array([[0, np.sqrt(3)/2, -np.sqrt(3)/2], [1, -0.5, -0.5]]))
    print(y)
    print(y.angles(1,2))
    print(y.check())
    print(y.nrow)
    print(y.constant())
    print(y.gramian())
    print(np.linalg.matrix_rank(y.gramian()))
    print(np.trace(y.gramian()))
    y_n = y.normalize()

    print(y_n.gramian())
    print(np.trace(y_n.gramian()))
    print(y_n.constant())
    print(y_n)
    print(y_n.frame_operator())
    print(y_n.check())
    print(y_n.constant())
    print(y.constant())
    print(np.trace(y_n.gramian()))
    print(y.frame_operator())
    print(y.gramian())
    z = y.complementary()
    print(z.frame_operator())

    print(z.normalize().gramian() + y.normalize().gramian())
    print(z)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
