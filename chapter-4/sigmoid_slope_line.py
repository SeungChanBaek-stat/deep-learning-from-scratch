### 왼쪽에 계단 함수, 오른쪽에 시그모이드 함수 그리기
### 추가로 각 함수의 기울기도 그리기

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 기울기 계산 함수
def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)

# 시그모이드 함수와 계단 함수 그리기
plt.plot(x, y1, label='Step Function')
plt.plot(x, y2, label='Sigmoid Function')
plt.ylim(-0.1, 1.1)  # y축 범위 지정

# 기울기를 계산할 두 점
points = [2, -3]

# 각 점에 대한 기울기 직선 그리기
for point in points:
    slope = sigmoid_gradient(point)  # 점에서의 기울기
    plt.plot(point, sigmoid(point), 'ro')  # 점 그리기
    # 기울기 직선의 시작과 끝 점 계산
    line_x = np.array([x[0], x[-1]])  # 직선을 그릴 x 값 범위
    line_y = slope * (line_x - point) + sigmoid(point)  # 직선의 y 값 계산
    plt.plot(line_x, line_y, 'r-')  # 기울기 직선 그리기

plt.legend()  # 범례 표시
plt.show()