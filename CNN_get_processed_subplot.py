import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


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
        h_conv1 = self.c1(x)
        h_pool1 = self.p1(h_conv1)
        h_conv2 = self.c2(h_pool1)
        h_pool2 = self.p2(h_conv2)
        y = self.flatten(h_pool2)
        y = self.f1(y)
        y = self.f2(y)
        y = self.f3(y)
        return y
        # return y

    def call_subgraph(self, x):
        with tf.name_scope('conv1'):
            h_conv1 = self.c1(x)
            h_pool1 = self.p1(h_conv1)
        with tf.name_scope('conv2'):
            h_conv2 = self.c2(h_pool1)
            h_pool2 = self.p2(h_conv2)
        with tf.name_scope('all_connection'):
            y0 = self.flatten(h_pool2)
            y1 = self.f1(y0)
            y2 = self.f2(y1)
            y = self.f3(y2)
        return h_conv1, h_pool1, h_conv2, h_pool2, y0, y1, y2


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

h_conv1, h_pool1, h_conv2, h_pool2, y0, y1, y2 = model.call_subgraph(x_test)

# save the subplot from the 1st training batch
np.save("./LeNet_subgraph_prediction/h_conv1.npy", h_conv1[0:32, :, :, :])
np.save("./LeNet_subgraph_prediction/h_pool1.npy", h_pool1[0:32, :, :, :])
np.save("./LeNet_subgraph_prediction/h_conv2.npy", h_conv2[0:32, :, :, :])
np.save("./LeNet_subgraph_prediction/h_pool2.npy", h_pool2[0:32, :, :, :])
np.save("./LeNet_subgraph_prediction/x_test.npy", x_test[0:32, :, :, :])
np.save("./LeNet_subgraph_prediction/y_test.npy", y_test[0:32])
# show info from h_conv1 to h_pool2
j = 0
plt.figure(figsize=(5, 5))
for i in range(10000):
    if (j is 10):
        break
    else:
        if (y_test[i, 0] == j):
            plt.subplot(10, 4, 4 * j + 1)
            plt.title('Type:{}'.format(j))
            plt.imshow(h_conv1[i, :, :, 1])
            plt.colorbar()

            plt.subplot(10, 4, 4 * j + 2)
            plt.title('Type:{}'.format(j))
            plt.imshow(h_pool1[i, :, :, 1])
            plt.colorbar()

            plt.subplot(10, 4, 4 * j + 3)
            plt.title('Type:{}'.format(j))
            plt.imshow(h_conv2[i, :, :, 1])
            plt.colorbar()

            plt.subplot(10, 4, 4 * j + 4)
            plt.title('Type:{}'.format(j))
            plt.imshow(h_pool2[i, :, :, 1])
            plt.colorbar()
            j += 1
plt.tight_layout()
plt.subplots_adjust(wspace=1, hspace=2)
plt.show()
