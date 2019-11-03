# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn import mixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn import mixture
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering



def hierarchy_classify(hierarchy_array, num_of_hierarchy):
    '''
    将聚好类的料面进行分组
    '''
    hierarchy = list()
    hierarchy.append(list())
    hierarchy.append(list())
    hierarchy.append(list())
    for i in range(num_of_hierarchy):
        for j in range(len(hierarchy_array)):
            if hierarchy_array[j] == i:
                hierarchy[i].append(j)
    return hierarchy





path = 'data_in/data4.txt'
source_data = np.loadtxt(path)
data = source_data[..., 20:]
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

# classed_hierarchy_gmm = hierarchy_classify(pred_gmm, 3)
# # classed_hierarchy_gmm_array = np.array(classed_hierarchy_gmm)
# print(classed_hierarchy_gmm_array)



# classed_hierarchy_kmeans = hierarchy_classify(pred_kmeans, 3)
# # classed_hierarchy_kmeans_array = np.array(classed_hierarchy_kmeans)
# print(classed_hierarchy_kmeans_array)

# classed_hierarchy_ward = hierarchy_classify(pred_ward, 3)
# # classed_hierarchy_ward_array = np.array(classed_hierarchy_ward)
# print(classed_hierarchy_ward_array)



