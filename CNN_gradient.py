#pre-processing for sensitivity analysis

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# LeNet
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.c2 = Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


model = LeNet5()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

record = model.fit(x_train, y_train,
                   batch_size=32,
                   epochs=1,
                   validation_data=(x_test, y_test),
                   validation_freq=5)

model.summary()
heatmap_array=[]
x = tf.convert_to_tensor(x_test[0:32, :, :, :])
for i in range(32):
    for j in range(10):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model.call(x)  # y=wx+b
            dy_dx = tf.square(tape.gradient(y[i][j], x))
            a = dy_dx.numpy()
            heatmap_array.append(a)

np.save("./LeNet_prediction/heatmap_matrix.npy",heatmap_array)