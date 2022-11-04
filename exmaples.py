import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from main import TightFrame

y = TightFrame(np.array([[0, np.sqrt(3) / 2, -np.sqrt(3) / 2], [1, -0.5, -0.5]]))
print(y)
print(y.angles(1, 2))
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
from main import entf

x = TightFrame(entf(8,3))
print(x.vector_norms())


print(x.normalize().frame_bounds())

print(y.frame_bounds())

x = TightFrame(np.array([[1,0,-0.5],[0,1,-0.5]]))

print(x.frame_bounds())


z = x.canonical_dual_frame()
print("-"*50)
print(y.canonical_dual_frame())


print(z.check())
print(z.gramian())
print(scipy.linalg.pinv(x.gramian()))


print(y)
print(y.canonical_dual_frame())
# y.plot_dual()

np.random.seed(1)
tf = TightFrame(np.random.randn(2,5))
# print(tf.constant())
# tf.normalize()
# np.zeros(tf.matrix.shape)
# print(tf.matrix)
tf.plot()
tf.plot_dual()
tf.plot_canonical()
plt.show()

print('------'*10)
z = tf.canonical_tight_frame()
z2 = tf.canonical_tight_frame(advanced=True)
print(z2.frame_operator())
print(z.frame_operator())
print(z.check())

print(z2.constant())
print(z.constant())

print(z.matrix)
print(z.canonical_dual_frame().matrix)

print(np.allclose(z.gramian() @ z.canonical_dual_frame().gramian() , z.canonical_tight_frame().gramian() ))