import tensorflow as tf
import numpy as np
from tensorflow import keras
# 模型保存路径
# save_path = './models/hello_model.h5'

# 训练的数据 y=3x+1
xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])
ys = np.array([-2, 1, 4, 7, 10, 13, 16, 19, 22])

# 新建模型
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  #序列训练模型 神经元 形状
model.compile(optimizer='sgd', loss='mean_squared_error') #随机梯度下降 均方误差
# 训练模型
model.fit(xs, ys, epochs=100)

# 保存模型
# model.save(save_path)

# 加载模型
# model = keras.models.load_model(save_path)

# model.summary()
print(model.predict([10]))