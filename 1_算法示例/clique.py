import numpy as np

# 选择聚类方法:clique 类
from pyclustering.cluster.clique import clique
# clique 可视化
from pyclustering.cluster.clique import clique_visualizer

# 构建训练教据
# d1=np.array([0, 2, 6.5, 8, 9])
# d2=np.array([0, 1, 2, 2.5, 9])

# d1=np.array([0, 1, 2, 4, 5, 4, 5, 8, 10, 11, 12])
# d2=np.array([0, 1, 2, 4, 5, 7, 8, 8, 10, 11, 12])

d1=np.array([0, 2, 3.5, 5,   5,   5,   9])
d2=np.array([0, 1, 2,   2.5, 4.5, 5.5, 9])
d3=np.array([0, 1, 4,   4.5, 5,   5.5, 9])

# d1=np.array([0, 0.9, 1, 9])
# d2=np.array([0, 3, 3, 6])
# d3=np.array([0, 2.1, 2.2, 3])

# data = np.array([d1])
# data = np.array([d1,d2])
data = np.array([d1,d2,d3])

data = data.T
data_M = np.array(data)
print(data_M)
# exit(0)

# 创建 CLIQUE 算法进行处理

# 定义每个维度中网格单元的数重
intervals = 3

# 密度阈值
threshold = 1

clique_instance=clique(data_M, intervals, threshold)
# print(clique_instance)
# exit(0)

#开始聚类过程并获得结果
clique_instance.process()
clique_cluster = clique_instance.get_clusters() # allocated clusters
print("clique_cluster:", clique_cluster)
print("Amount of clusters:", len(clique_cluster))
# exit(0)

#被认为是异常值的点(噪点)
noise = clique_instance.get_noise()
print("noise:", noise)
# exit(0)

# CLIOUE形成的网格单元
cells = clique_instance.get_cells()
print("cells:", cells)
print("Amount of cells:", len(cells))


#显示由算法形成的网格
clique_visualizer.show_grid(cells, data_M)
# #显示聚类结果
# clique_visualizer.show_clusters(data_M, clique_cluster, noise)# show clustering results