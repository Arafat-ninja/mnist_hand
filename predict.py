import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('mnist.npz')#读取数据集  keras是接口
x_train = tf.keras.utils.normalize(x_train, axis=1)#划分训练集
x_test = tf.keras.utils.normalize(x_test, axis=1)#划分测试集

model = tf.keras.models.load_model('Mnist.h5')     #加载你训练好的数据集

#单张预测
i = 16
plt.imshow(x_test[i],cmap=plt.cm.binary)        #颜色进行选取


predictions = model.predict(x_test)
print(x_test[i].shape)
#print(np.argmax(predictions[i]))       #取最大值

plt.rcParams['font.sans-serif']=['SimHei']       #解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.title('预测值为：%d'%(np.argmax(predictions[i])))
plt.show()