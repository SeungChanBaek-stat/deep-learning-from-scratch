import sys, os
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()
    # plt.imshow(np.array(pil_img))
    # plt.show()
    
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    
img = x_train[0]
label = t_train[0]

print(type(img)) # <class 'numpy.ndarray'>
print(label) # 5

print(img.shape)          # (784,)
img = img.reshape(28, 28) # 원래 이미지의 모양으로 변형
print(img.shape)          # (28, 28)

img_show(img)