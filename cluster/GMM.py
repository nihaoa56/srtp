# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn import mixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn import mixture
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA   #pca降维工具



path = 'data_in/data4.txt'
source_data = np.loadtxt(path)
data = source_data[..., 21:]
print(data.shape)
# 高斯混合聚类
gmm=mixture.GaussianMixture(n_components=3)
gmm.fit(data)
pred_gmm = gmm.predict(data)
print("高斯混合聚类结果：")
print('gmm:', np.unique(pred_gmm))
print(pred_gmm)
# Kmeans聚类
kmeans=KMeans(n_clusters=3)#聚类个数
kmeans.fit(data)#训练
pred_kmeans = kmeans.labels_  #每个样本所属的类
print("Kmeans聚类结果 :")
print('kmeans:', np.unique(kmeans.labels_))
print(pred_kmeans)

ward = AgglomerativeClustering(n_clusters=3).fit(data)
pred_ward = ward.fit_predict(data)
print("层次聚类结果 :")
print('层次:', np.unique(pred_ward))
print(pred_ward)


# pca降维
pca_machine = PCA(n_components=3,whiten=True)
pca_machine.fit(data)

pca_result = pca_machine.fit_transform(data)
# print(pca_result)
# print(pca_result[:,0])
pca_X = pca_result[:, 0]
pca_Y = pca_result[:, 1]
pca_Z = pca_result[:,2]

class_one = np.arange(3) #矩阵无效初始化
class_two = np.arange(3)
class_three = np.arange(3)



for i in range(len(pred_kmeans)):
    if pred_kmeans[i] == 0:
        class_one = np.vstack((class_one,pca_result[i]))
    elif pred_kmeans[i] == 1:
        class_two = np.vstack((class_two,pca_result[i]))
    elif pred_kmeans[i] == 2:
        class_three = np.vstack((class_three, pca_result[i]))
        
# 矩阵处理
class_one = np.delete(class_one, 0, 0)
class_two = np.delete(class_two, 0, 0)
class_three = np.delete(class_three, 0, 0)

class_one_X = class_one[:, 0]
class_one_Y = class_one[:, 1]
class_one_Z = class_one[:, 2]

class_two_X = class_two[:, 0]
class_two_Y = class_two[:, 1]
class_two_Z = class_two[:, 2]

class_three_X = class_three[:, 0]
class_three_Y = class_three[:, 1]
class_three_Z = class_three[:, 2]



ax = plt.subplot(111, projection='3d') # 创建一个三维的绘图工程
# 将数据点分成三部分画，在颜色上有区分度
ax.scatter(pca_X[:15], pca_Y[:15], pca_Z[:15], c='y') # 绘制数据点
ax.scatter(pca_X[15:30], pca_Y[15:30], pca_Z[15:30], c='r')
ax.scatter(pca_X[30:50], pca_Y[30:50], pca_Z[30:50], c='g')



# ax.scatter(class_one_X, class_one_Y, class_one_Z, c='y') # 绘制数据点
# ax.scatter(class_two_X, class_two_Y,class_two_Z , c='r')
# ax.scatter(class_three_X, class_three_Y, class_three_Z, c='g')

ax.set_zlabel('Z') # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.savefig('d.jpg')
plt.show()

# classed_hierarchy_gmm = hierarchy_classify(pred_gmm, 3)
# # classed_hierarchy_gmm_array = np.array(classed_hierarchy_gmm)
# print(classed_hierarchy_gmm_array)



# classed_hierarchy_kmeans = hierarchy_classify(pred_kmeans, 3)
# # classed_hierarchy_kmeans_array = np.array(classed_hierarchy_kmeans)
# print(classed_hierarchy_kmeans_array)

# classed_hierarchy_ward = hierarchy_classify(pred_ward, 3)
# # classed_hierarchy_ward_array = np.array(classed_hierarchy_ward)
# print(classed_hierarchy_ward_array)



