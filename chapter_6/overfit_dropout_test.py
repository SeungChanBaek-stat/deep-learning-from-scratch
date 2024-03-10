# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import math
import time

start = time.time()




(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.5

# 최적화 알고리즘 및 모수 설정 ========================
optimizer_dict = ['SGD', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSprop', 'Adam']

optimizer = 'Adam'

# 시간 측정용 딕셔너리 설성 ============================
time_dict = {}

# ====================================================

idx = 1    
    
plt.figure(figsize=(10, 10))  # 여기로 이동: 모든 그래프를 포함할 새로운 Figure 생성


for item in optimizer_dict:
    
    # 시간 측정 시작
    start = time.time()
    optimizer = item

    if optimizer == 'Adam':
        optimizer_param = {'learning_rate': 0.001}
    else :
        optimizer_param = {'learning_rate': 0.01}
        
    

    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=301, mini_batch_size=100,
                    optimizer=optimizer, optimizer_param=optimizer_param, verbose=True)
    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list
    
    # 시간 측정 끝
    end = time.time()
    print(f"학습 시간 : {end - start:.5f} sec")
    time_dict[item] = end - start

    # 그래프 그리기==========
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.subplot(3, 2, idx)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    if use_dropout == True :
        plt.title(f"dropout_ratio : {dropout_ratio}, optimizer : {optimizer}")
    else :
        plt.title(f"No dropout, optimizer : {optimizer}")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    idx += 1

for key in time_dict.keys():
        print(f"{key} 학습 시간 : {time_dict[key]:.5f} sec")    
    
plt.show()
