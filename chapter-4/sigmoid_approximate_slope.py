# 그림 4-5 
# -*- coding: utf-8 -*-
from matplotlib import rc
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fp = fm.FontProperties(fname="c:/Windows/Fonts/NGULIM.ttf")
# MacOS 경우
#fp = fm.FontProperties(fname="/Users/plusjune/Library/Fonts/NanumGothic.ttf")
rc('font', family=fp.get_name())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_tangent(x): # 접선 ax+b에서 a,b 값을 리턴
    return sigmoid_diff(x), sigmoid(x) - sigmoid_diff(x) * x

x = np.arange(-6.0, 6.0, 0.1)
y1 = sigmoid(x)
a2, b2 = sigmoid_tangent(0)
y2 = a2 * x + b2
a3 = (sigmoid(2.5) - sigmoid(0)) / 2.5
y3 = a3 * x + b2

plt.plot(x, y1, label='y=f(x)')
plt.plot(x, y2, color='black', label='진정한 접선')
plt.plot(x, y3, color='green', label='근사로 구한 접선')
xv = np.arange(-0.1, 0.5, 0.01)
plt.text(-0.2,0,"x")
plt.plot(np.array([0 for _ in range(xv.size)]), xv, 'k--')
xhv = np.arange(-0.1, sigmoid(2.5), 0.01)
plt.text(2,0,"x+h")
plt.plot(np.array([2.5 for _ in range(xhv.size)]), xhv, 'k--')
plt.scatter([0],[b2],color='red')

plt.ylim(-0.1,1.1)
plt.xlim(-4,4)
plt.legend(loc='upper center')
plt.show()