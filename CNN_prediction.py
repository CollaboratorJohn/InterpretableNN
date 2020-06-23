from PIL import Image
import numpy as np
import tensorflow as tf
import os

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model_save_path = './checkpoint/LeNet_cifar.ckpt'
#use LeNet
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5),activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5),activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)
path="./LeNet_prediction"
result_tensor = model.predict(x_test[0:32,:,:,:])
result=[]

for i in range(32):
    result.append(np.argmax(result_tensor[i]))

'''
np.save(path+"LeNet_prediction.npy",result)
prediction = tf.argmax(result, axis=1)
#tf.print(prediction)'''
sensitivity=[]

for i in range(32):
    x_=tf.convert_to_tensor(x_test[0:32,:,:,:])
    with tf.GradientTape() as g:
        g.watch(x_)
        y=model.predict(x_)
        #dy_dx=g.gradient(y,x_)


#np.save(path+"Sensitivity_Data.npy",sensitivity)