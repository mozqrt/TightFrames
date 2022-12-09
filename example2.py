import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from main import TightFrame

y = TightFrame(np.array([[0, 0, 1], [1, 0, 0], [-0.5, -np.sqrt(3) / 2, 0], [-0.5, np.sqrt(3) / 2, 0]]))
print(y.check())
print(y.matrix)
print(y.angles(0,1))
#
z = TightFrame(np.array([[0, 0, 1], [np.sqrt(3) / 2, 0, -0.5], [-np.sqrt(3) / 4, -3/4, -1/2], [-np.sqrt(3) / 4, 3/4, -1/2]]))
print(z.matrix)
print(z.check())
print(z.angles(1, 3))
# print(z.)


x = TightFrame(np.array([[0, 0, 1], [np.sqrt(5/6), 0, np.sqrt(1/6)], [-np.sqrt(5/6), 0, np.sqrt(1/6)], [0, np.sqrt(5/6), np.sqrt(1/6)],  [0, -np.sqrt(5/6), np.sqrt(1/6)]]))
print(x.check())
print(x.angles(1,2))
print(x.gramian())


def Welch_inequality(frame):
    oper = frame.frame_operator()
    lhs = np.trace(oper @ oper)
    rhs = (np.trace(oper)) ** 2 / frame.nrow
    return(lhs, rhs)

rand_frame = TightFrame(np.random.randn(400, 3))
print(rand_frame.matrix)
print(Welch_inequality(rand_frame))


orth = TightFrame(np.array([[1,0,0], [0,1,0], [0,0,1]]))
print(Welch_inequality(y))
print(y.check())
print(z.frame_operator())


merc = TightFrame(np.array([[0,1], [-np.sqrt(3)/ 2, -0.5, ], [np.sqrt(3)/ 2, -0.5]]))
print(merc.matrix)
print(merc.nrow)
print(merc.frame_operator())
print(merc.gramian())
print(Welch_inequality(merc))
merc.plot()
plt.show()


print(Welch_inequality(rand_frame))