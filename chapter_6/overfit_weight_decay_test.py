# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optim import *
from common.weight_decay_test import weight_decay_test

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]
        

# weight decay（가중치 감쇠） 설정 =======================
# weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
# weight_decay_lambda = 0.1
# ====================================================

optimizer_dict = {'SGD': SGD(), 'Momentum': Momentum(), 'AdaGrad': AdaGrad(), 'RMSprop': RMSprop(), 'Adam': Adam()}

penalization = 'L1' # L1 or L2

weight_decay_lambda_list = [0, 0.01, 0.1, 1] # lambda 값 리스트


for key in optimizer_dict.keys():
    optimizer = optimizer_dict[key]
    # optimizer = SGD() # 학습알고리즘 선택 : SGD, Momentum, AdaGrad, RMSprop, Adam
    weight_decay_lambda_list = weight_decay_lambda_list
    idx = 1

    plt.figure(figsize=(10, 10))  # 여기로 이동: 모든 그래프를 포함할 새로운 Figure 생성

    for i in weight_decay_lambda_list:
        # 테스트 진도
        print(f"===========| weight_decay_lambda : {i}, optimizer : {optimizer.__class__.__name__}, penalization : {penalization} |===========")
        # 초기화
        decay_test_lambda = weight_decay_test(x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test,
                                              weight_decay_lambda=i, optimizer=optimizer, penalization=penalization)
        # 학습
        decay_test_lambda.train()
        train_acc_list, test_acc_list = decay_test_lambda.train()
        max_epochs = decay_test_lambda.max_epochs
        
        # 그래프 그리기==========
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(max_epochs)
        plt.subplot(2, 2, idx)
        plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
        plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
        plt.title(f"lambda : {decay_test_lambda.weight_decay_lambda}, optimizer : {optimizer.__class__.__name__}")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        idx += 1

    # plt.show()
    plt.savefig(f'{optimizer.__class__.__name__}_weight_decay_test.png', dpi=200)

