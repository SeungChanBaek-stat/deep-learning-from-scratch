### f(x_0, x_1) = x_0^2 + x_1^2 의 그래프
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

x0 = np.arange(-3, 3, 0.1)
x1 = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(x0, x1)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()
plt.savefig('C:\\Users\\SAMSUNG\\Downloads\\deep-learning-from-scratch-series\\Part-1\\chapter-4\\images\\fig 4-7.png')