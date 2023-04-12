import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # 序列训练模型
model.compile(optimizer='sgd', loss='mean_squared_error')

model.load_weights(checkpoint_path)

xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])
ys = np.array([-2, 1, 4, 7, 10, 13, 16, 19, 22])

value1 = model.predict([1])

model.fit(xs, ys, epochs=1500)
value2 = model.predict([1])
print(value1, value2)