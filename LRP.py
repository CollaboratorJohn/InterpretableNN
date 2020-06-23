import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import log
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import os

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
    #获得子图
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

class LRP_Analysis:
    def __init__(self):
        self.alpha=2
    #层间传播关系
    def backdrop_dense(self,activation,kernel,bias,relevance):
        W_p=tf.maximum(0,kernel)
        b_p=tf.maximum(0,bias)
        z_p=tf.matmul(activation,W_p)+b_p
        s_p=relevance/z_p
        c_p=tf.matmul(s_p,tf.transpose(W_p))

        W_n=tf.minimum(0,kernel)
        b_n=tf.minimum(0,bias)
        z_n=tf.matmul(activation,W_n)+b_n
        s_n=relevance/z_n
        c_n=tf.matmul(s_n,tf.transpose(W_n))
        return activation*(self.alpha*c_p+(1-self.alpha)*c_n)
    #计算输出交叉熵
    def output_cross_entropy(self,out,out_ref):
        y=[]
        standard=tf.one_hot(indices=out_ref,depth=10,axis=1)
        standard=standard.numpy()
        standard=standard.squeeze()
        for i in range(10):
            entropy=-standard[i]*log(out[i])
            y.append(entropy)
        y=np.array(y)
        y=y.reshape((1,10))
        return y
    #计算dropout后的输出
    def prediction_after_drop(self,yin,kernel1,kernel2,bias1,bias2,drop1,drop2):
        #将影响小的神经元参数置零以实现dropout
        kernel1=kernel1.numpy()
        kernel2=kernel2.numpy()
        for i in drop1:
            kernel1[i,:]=np.zeros(84,)
        for j in drop2:
            kernel2[j,:]=np.zeros(10,)
        kernel1=tf.convert_to_tensor(kernel1)
        kernel2=tf.convert_to_tensor(kernel2)
        #全连接计算
        y1=tf.matmul(yin,kernel1)+bias1
        y2=tf.nn.sigmoid(y1)
        y3=tf.matmul(y2,kernel2)+bias2
        y4=tf.nn.softmax(y3)
        return y4

#在基准模型上预测
model=LeNet5()
#载入断点
checkpoint_save_path="./checkpoint/LeNet_cifar_class.ckpt"

if(os.path.exists(checkpoint_save_path + '.index')):
    print('loaded successfully!')
    model.load_weights(checkpoint_save_path)
#计算准确率
result_tensor = model.predict(x_test[0:1000,:,:,:])
result=[]
accuracy=0
for i in range(1000):
    result.append(np.argmax(result_tensor[i]))
    if(np.argmax(result_tensor[i])==y_test[i]):
        accuracy+=1

# 选取第一个测试batch演示
[h_conv1, h_pool1, h_conv2, h_pool2, y0, y1, y2] = model.call_subgraph(x_test[0:1000, :, :, :])
y = model.call(x_test[0:1000, :, :, :])
#读取需要的训练数据
for v in model.trainable_variables:
    if str(v.name) == "le_net5/dense_2/kernel:0":
        kernel2 = v.numpy()
    if str(v.name) == "le_net5/dense_2/bias:0":
        bias2=v.numpy()
    if str(v.name) == "le_net5/dense_1/kernel:0":
        kernel1 = v.numpy()
    if str(v.name) == "le_net5/dense_1/bias:0":
        bias1 = v.numpy()

LRP=LRP_Analysis()
R=tf.zeros((1,10))#输出层神经元权重
R_Dense2=tf.zeros((1,84))#倒数第一层神经元权重
R_Dense1=tf.zeros((1,120))#倒数第二层神经元权重
for i in range(1000):
    #用交叉熵的方式运算输出层关联R
    R+=LRP.output_cross_entropy(y[i,:],y_test[i,:])
    #将断点训练数据转化为张量
    kernel2=tf.convert_to_tensor(kernel2)
    bias2=tf.convert_to_tensor(bias2)
    kernel1=tf.convert_to_tensor(kernel1)
    bias1=tf.convert_to_tensor(bias1)
    #运算倒数第一层全连接层关联
    yy=np.array(y2[i,:])
    yy=yy.reshape((1,84))
    activation=tf.convert_to_tensor(yy)
    R=tf.convert_to_tensor(R)
    R=tf.cast(R, dtype=tf.float32)
    R_Dense2+=LRP.backdrop_dense(activation,kernel2,bias2,R)
    #运算倒数第二层全连接层关联
    yy=np.array(y1[i,:])
    yy=yy.reshape((1,120))
    activation=tf.convert_to_tensor(yy)
    R_Dense1+=LRP.backdrop_dense(activation,kernel1,bias1,R_Dense2)
'''
Dense2_neg=tf.where(abs(R_Dense2)>10000).numpy()
Dense2_neg=Dense2_neg[:,1]
'''
Dense1_neg=tf.where(abs(R_Dense1)<1000).numpy()
Dense1_neg=Dense1_neg[:,1]
Dense2_neg=np.random.randint(low=0,high=83,size=20)
'''
plt.subplot(1,3,1)
plt.bar(np.arange(0,10),(R.numpy()).reshape((10,)))
plt.subplot(1,3,2)
plt.bar(np.arange(0,84),(R_Dense2.numpy()).reshape((84,)))
plt.subplot(1,3,3)
plt.bar(np.arange(0,120),(R_Dense1.numpy()).reshape((120,)))
plt.show()
'''
#测试dropout之后的结果以及准确率
accuracy2=0
for i in range(1000):
    yy = np.array(y1[i, :])
    yy = yy.reshape((1, 120))
    #输出dropout的预测结果以及实际结果
    print(np.argmax(LRP.prediction_after_drop(yy,kernel1,kernel2,bias1,bias2,Dense1_neg,Dense2_neg)),
          y_test[i])
    if(np.argmax(LRP.prediction_after_drop(yy,kernel1,kernel2,bias1,bias2,Dense1_neg,Dense2_neg))==y_test[i]):
        accuracy2+=1

print(accuracy)
print(accuracy2)