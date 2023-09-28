import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch as torch




def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    data = ts.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        if label[i]==0:
            plt.text(data[i, 0], data[i, 1], str(int(label[i])), color=plt.cm.Set1(label[i]),
                    fontdict={'size': 3})
        else:
            plt.text(data[i, 0], data[i, 1], str(int(label[i])), color=plt.cm.Set1(label[i]),
                    fontdict={'size': 7})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    return fig

def plot_embedding_heatmap(data, label, title, mask):
    # 去除一下outliers  看下数据的分布的情况
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    data = ts.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig, ax = plt.subplots(figsize=(8,5), dpi = 600)
    # 去除label中的离群值 再重新画 暂定为排序从5%-95%
    temp = np.sort(label)
    lower_value = temp[int(len(label)*0.05)]
    upper_value = temp[int(len(label)*0.95)]
    new_mask = (label>lower_value)*(label<upper_value)
    scatter = ax.scatter(data[:,0][new_mask].squeeze(), data[:,1][new_mask].squeeze(), c=label[new_mask], cmap='coolwarm', alpha=0.7,)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    legend2 = ax.legend(handles, labels, loc="upper right", title="Colors")
    plt.title(title, fontsize=14)
    return fig


def cal_distance(data, mask):
    two_distances = np.dot(data, data.T) # [1026,1026]
    weight_mask = mask.reshape(-1,1)
    # 这里这两个阈值 还需要待定
    hard_mask = np.absolute(np.dot(weight_mask, weight_mask.T))
    hard_avg = (hard_mask*two_distances).sum()/(hard_mask.sum()) # 这里不能直接取mean
    easy_avg = ((1-hard_mask)*two_distances).sum()/((1-hard_mask).sum())
    all_avg = two_distances.mean()
    return hard_avg, easy_avg, all_avg


