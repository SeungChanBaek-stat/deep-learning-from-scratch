# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.multi_layer_net import MultiLayerNet


class weight_decay_test:
    def __init__(self, x_train, t_train, x_test, t_test, weight_decay_lambda, optimizer, penalization='L2'):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.weight_decay_lambda = weight_decay_lambda
        self.penalization = penalization
        self.network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=self.weight_decay_lambda, penalization=self.penalization)
        self.optimizer = optimizer
        self.optimizer_name = optimizer.__class__.__name__

        self.max_epochs = 201
        self.train_size = self.x_train.shape[0]
        self.batch_size = 100

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.epoch_cnt = 1
    
    def train(self):
        for i in range(1000000000):
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)

            if i % self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                # # 차원 불일치 오류 검증용 코드
                # print(f"train_acc_list 길이 : {len(self.train_acc_list)}, epoch_cnt : {self.epoch_cnt}")

                print("epoch:" + str(self.epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

                self.epoch_cnt += 1
                if self.epoch_cnt >= self.max_epochs:
                    break
        return self.train_acc_list, self.test_acc_list