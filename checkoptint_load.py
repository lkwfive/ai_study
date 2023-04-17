import tensorflow as tf
import numpy as np
from tensorflow import keras

checkpoint_path = "training/cp.ckpt"

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # 序列训练模型
model.compile(optimizer='sgd', loss='mean_squared_error')

model.load_weights(checkpoint_path)

xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
ys = np.array([-2, 1, 4, 7, 10, 13, 16, 19, 22, 25])

value1 = model.predict([9])

model.fit(xs, ys, epochs=1000)
value2 = model.predict([9])
print(value1, value2)