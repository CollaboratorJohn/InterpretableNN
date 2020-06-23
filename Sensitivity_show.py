import numpy as np
import matplotlib.pyplot as plt
path="./LeNet_prediction/"
heatmap_matrix=np.load(path+"heatmap_matrix.npy")
#sensitivity=np.load(path+"Sensitivity_Data.npy")
heatmap_info=np.zeros((10,32,32,3))
for i in range(10):
    for j in range(32):
        heatmap_info[i]+=heatmap_matrix[10*j+i,j]

plt.figure(figsize=(5,5))
for i in range(5):
    plt.subplot(5,2,2*i+1)
    plt.imshow(heatmap_info[2*i,:,:,2],cmap='hot',alpha=0.5)
    plt.title('Type:{}'.format(2*i))
    plt.colorbar()

    plt.subplot(5,2,2*i+2)
    plt.imshow(heatmap_info[2*i+1,:,:,2],cmap='hot',alpha=0.5)
    plt.title('Type:{}'.format(2*i+1))
    plt.colorbar()
#readjust and show the whole picture
plt.tight_layout()
plt.subplots_adjust(wspace=1, hspace=1)
plt.show()