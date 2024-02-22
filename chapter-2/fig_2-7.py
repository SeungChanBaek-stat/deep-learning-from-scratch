import numpy as np
import matplotlib.pyplot as plt
import matplotlib

x1 = np.arange(-1, 3, 0.1)
x2 = -x1 + 0.5

plt.axvline(x=0, color = 'b')  # draw x =0 axes
plt.axhline(y=0, color = 'b')   # draw y =0 axes
plt.xlabel("X1") # x축 이름
plt.ylabel("X2") # y축 이름

plt.scatter([0,1],[0,1],marker='o',color='r')
plt.scatter([1,0],[0,1],marker='^',color='r')
plt.show()