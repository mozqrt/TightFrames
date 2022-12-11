import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from main import TightFrame

np.random.seed(1)
x = TightFrame(np.random.rand(5,3))
print(x.matrix)
print("-" * 50)
print(x.canonical_tight_frame())
print("-" * 50)

print(x.canonical_tight_frame(svd=False))


# print("-" * 50)
# U, S, V = la.svd(x.matrix)
# print(U)
# print(S)
# print(V)
# print(np.diag(1/S))
# print(U @ np.diag(1/S) @ V.T)
