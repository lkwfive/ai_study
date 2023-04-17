import tensorflow as tf
import numpy as np
from tensorflow import keras

checkpoint_path = "training/cp.ckpt"

# save_weights_only:保存权重 verbose:显示日志
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])
ys = np.array([-2, 1, 4, 7, 10, 13, 16, 19, 22])

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # 序列训练模型
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=10, callbacks=[cp_callback])
