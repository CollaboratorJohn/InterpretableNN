import cv2
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt

def readjust(matrix):
    subimg = cv2.resize(matrix, (32, 32))
    return subimg


'''
filter name:使用某个卷积核生成的CNN子图
type_sorted:对于分类的第k类图像
'''
def calc_loss_for_onetype(filter_name, type_sorted, sub_graph, filter_graph, sub_graph_num, filter_graph_num):
    Z_T = [0] * filter_graph_num
    p_x_T = [[0] * sub_graph_num] * filter_graph_num
    p_T = []
    p_x = [0] * sub_graph_num
    loss_array = []
    for i in range(filter_graph_num):
        p_T.append(exp((np.sum(filter_graph[i]))))
        for j in range(sub_graph_num):
            temp_graph = np.multiply(sub_graph[j], filter_graph[i])
            temp_trace = np.sum(temp_graph)
            p_x_T[i][j] = temp_trace
            Z_T[i] += exp(temp_trace)

    for i in range(filter_graph_num):
        for j in range(sub_graph_num):
            p_x_T[i][j] = p_x_T[i][j] / Z_T[i]

    for j in range(sub_graph_num):
        for i in range(filter_graph_num):
            p_x[j] += p_T[i] * p_x_T[i][j]

    for i in range(filter_graph_num):
        loss = 0
        for j in range(sub_graph_num):
            loss += p_T[i] * p_x_T[i][j] * log(p_x_T[i][j] / p_x[j])
            #loss +=  p_x_T[i][j] * log(p_x_T[i][j] )
        loss_array.append(loss)

    return filter_name, type_sorted, loss_array


h_conv1 = np.load('./LeNet_subgraph_prediction/h_conv1.npy')
h_pool1 = np.load('./LeNet_subgraph_prediction/h_pool1.npy')
h_conv2 = np.load('./LeNet_subgraph_prediction/h_conv2.npy')
h_pool2 = np.load('./LeNet_subgraph_prediction/h_pool2.npy')
y_test = np.load('./LeNet_subgraph_prediction/y_test.npy')
T = np.load('./LeNet_prediction/heatmap_matrix.npy')

norm_conv1_core1 = []
for i in range(32):
    norm_conv1_core1.append(readjust(h_conv1[i, :, :, 0]))

norm_conv1_core1=np.array(norm_conv1_core1)
plt.subplot(11,1,1)
plt.imshow(norm_conv1_core1[3,:,:])
plt.colorbar()
for k in range(10):
    plt.subplot(11,1,k+2)
    plt.imshow(T[10 * 3 + k, 3, :, :, 0])
    plt.colorbar()
plt.show()

sub_graph_num = 0
filter_graph_num = 0
sub_graph = []
filter_graph = []
for i in range(32):
    if (y_test[i] == 0):
        sub_graph_num += 1
        filter_graph_num += 10
        sub_graph.append(norm_conv1_core1[i])
        for j in range(10):
            filter_graph.append(T[10 * i + j, i, :, :, 1])
        break

[filter_name, type_sorted, loss_array]=calc_loss_for_onetype('conv1_core1_R', 'type0', sub_graph, filter_graph, sub_graph_num, filter_graph_num)
print("%s:%s:loss_sequence:"%(filter_name,type_sorted))
print(loss_array)