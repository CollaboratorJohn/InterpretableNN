import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.python.framework import ops

# np.set_printoptions(threshold=inf)
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# AlexNet
class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')
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
    def call(self,x):
        x=self.c1(x)
        x=self.p1(x)
        x=self.c2(x)
        x=self.p2(x)
        x=self.flatten(x)
        x=self.f1(x)
        x=self.f2(x)
        y=self.f3(x)
        return y
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)
        self.flatten = Flatten()
        self.f1 = Dense(120, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

model = LeNet5()
'''
model = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5),activation='sigmoid'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=6, kernel_size=(5, 5),activation='sigmoid'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(120, activation='sigmoid'),
    Dense(84, activation='sigmoid'),
   Dense(10, activation='softmax')
])
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#断点续训

path="./checkpoint/LeNet_cifar_class.ckpt"
if os.path.exists(path+".index"):
    print("------load the model:-------")
    model.load_weights(path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 save_best_only=True)

record=model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test),
          validation_freq=5,
          callbacks=[cp_callback])

model.summary()

print(model.trainable_variables)
txt_path='./checkpoint/LeNet_cifar10_class_weights.txt'
file=open(txt_path,'w')
file.write("LeNet_cifar10_class information:"+'\n')
#分别是一个权重矩阵，一个偏置向量，一个学习率和计步器
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()

acc=record.history['sparse_categorical_accuracy']
val_acc=record.history['val_sparse_categorical_accuracy']
loss=record.history['loss']
val_loss=record.history['val_loss']
#plot
plt.subplot(1,2,1)
plt.plot(acc,label='training Accuracy')
plt.plot(val_acc,label='validation Accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='training Loss')
plt.plot(val_loss,label='validation Loss')
plt.title('training and validation Loss')
plt.legend()

plt.show()
