## optimizers
import numpy as np

# Stochastic Gradient Descent
class SGD:
    def __init__(self, learning_rate=None):
        if learning_rate is None:
            self.lr = 0.01
        else:
            if learning_rate <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.lr = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
# Momentum            
class Momentum:
    def __init__(self, learning_rate=None, momentum=None):
        # 학습률 설정
        if learning_rate is None:
            self.lr = 0.01
        else:
            if learning_rate <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.lr = learning_rate
        
        # 모멘텀 계수 설정                
        if momentum is None:
            self.momentum = 0.9
        else:
            if momentum <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.momentum = momentum
        
        # 속도 초기화                
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items(): # params 차원만큼 v 초기화
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = (self.momentum * self.v[key]) + (self.lr * grads[key])
            params[key] -= self.v[key]
            
            
# AdaGrad          
class AdaGrad:
    def __init__(self, learning_rate=None):
        if learning_rate is None:
            self.lr = 0.01
        else:
            if learning_rate <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.lr = learning_rate
        
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items(): # params 차원만큼 h 초기화
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 기울기 제곱 원소별로 누적
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 가중치 원소별로 업데이트
            
            
# RMSprop
class RMSprop:
    def __init__(self, learning_rate=None, moving_rate=None):
        if learning_rate is None:
            self.lr = 0.01
        else:
            if learning_rate <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.lr = learning_rate
                
        if moving_rate is None:
            self.mr = 0.9
        else:
            if moving_rate <= 0:
                raise ValueError('rho must be positive value')
            else:
                self.mr = moving_rate
        
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items(): # params 차원만큼 h 초기화
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            # grads[key] = np.clip(grads[key], -2.0, 2.0) # 그라디어트 클리핑
            self.h[key] *= self.mr
            self.h[key] += (1 - self.mr) *(grads[key] * grads[key]) # 기울기 제곱 원소별로 누적
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8) # 가중치 원소별로 업데이트
            

# Adam
class Adam:
    def __init__(self, learning_rate=None, beta_1=None, beta_2=None):
        if learning_rate is None:
            self.lr = 0.001
        else:
            if learning_rate <= 0:
                raise ValueError('learning_rate must be positive value')
            else:
                self.lr = learning_rate
                
        if beta_1 is None:
            self.b1 = 0.9
        else:
            if beta_1 <= 0:
                raise ValueError('beta_1 must be positive value')
            else:
                self.b1 = beta_1
        
        if beta_2 is None:
            self.b2 = 0.999
        else:
            if beta_2 <= 0:
                raise ValueError('beta_1 must be positive value')
            else:
                self.b2 = beta_2
                
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items(): # params 차원만큼 m 초기화
                self.m[key] = np.zeros_like(val)
                
        if self.v is None:
            self.v = {}
            for key, val in params.items(): # params 차원만큼 m 초기화
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        b1_t = self.b1 ** self.iter
        b2_t = self.b2 ** self.iter
                
        for key in params.keys():
            # grads[key] = np.clip(grads[key], -2.0, 2.0) # 그라디어트 클리핑
            self.m[key] *= self.b1
            self.m[key] += (1 - self.b1) *(grads[key]) # 기울기 제곱 원소별로 누적
            self.v[key] *= self.b2
            self.v[key] += (1 - self.b2) *(grads[key] * grads[key]) # 기울기 제곱 원소별로 누적
            
            m_hat = self.m[key] / (1 - b1_t  )
            v_hat = self.v[key] / (1 - b2_t  )
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8) # 가중치 원소별로 업데이트
            
            
            
            
            
class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""
    # NAG는 모멘텀에서 한 단계 발전한 방법이다. (http://newsight.tistory.com/224)
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]