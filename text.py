import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('mnist.npz')#读取数据集  keras是接口
x_train = tf.keras.utils.normalize(x_train, axis=1)#划分训练集
x_test = tf.keras.utils.normalize(x_test, axis=1)#划分测试集

model = tf.keras.models.load_model('Mnist1.h5')
model.summary()
#单张预测

# i = Image.open(r'C:\Users\KFC\Desktop\2.png')
# i = i.convert('L')
#
# a = i.resize((28,28))
#
# a = np.array(a) / 255.
# a=np.reshape(a,[1,28,28])
# # a = a.reshape((784))
# plt.imshow(i)        #颜色进行选取


img=cv2.imread(r'C:\Users\KFC\Desktop\1.png',0)/255.
# img=cv2.resize(img,(28,28))
img=np.reshape(img,[1,28,28])
img=img.astype(np.float64)
predictions = model.predict(img)
#print(i.shape)
#print(np.argmax(predictions[i]))       #取最大值
print(np.argmax(predictions))
# plt.rcParams['font.sans-serif']=['SimHei']       #解决中文显示问题
# plt.rcParams['axes.unicode_minus'] = False
# plt.title('预测值为：%d'%(np.argmax(predictions)))
# plt.show()