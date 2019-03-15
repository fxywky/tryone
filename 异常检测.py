#！/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat("data/ex8data1.mat")
X = data["X"]   # (307, 2)

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(X[:, 0], X[:, 1])
# plt.show()

# 这是一个非常紧密的聚类，几个值远离了聚类。 在这个简单的例子中，
# 这些可以被认为是异常的。 为了弄清楚，我们正在为数据中的每个特征估计高斯分布。
# 为此，我们将创建一个返回每个要素的均值和方差的函数。
def estimate_gaussian(X):
    mu = X.mean(axis=0)        # 每一列的均值
    sigma = X.var(axis=0)

    return mu, sigma


mu, sigma = estimate_gaussian(X)   # (2,)   (2,)
# 现在我们有了我们的模型参数，我们需要确定概率阈值，这表明一个样本应该被认为是一个异常。
# 为此，我们需要使用一组标记的验证数据（其中真实异常样本已被标记），
# 并在给出不同阈值的情况下，对模型的性能进行鉴定。
Xval = data['Xval']   # (307, 2)
yval = data['yval']   # (307, 1)

# 我们还需要一种计算数据点属于正态分布的概率的方法。 幸运的是SciPy有这个内置的方法
from scipy import stats
dist = stats.norm(mu[0], sigma[0])
pro = dist.pdf(15)     # 输入x，返回概率密度函数   0.1935875044615038

# 我们还可以将数组传递给概率密度函数，并获得数据集中每个点的概率密度。
pro1 = dist.pdf(X[:, 0])[0:50]      # (50,)

# 我们计算并保存给定上述的高斯模型参数的数据集中每个值的概率密度。
p = np.zeros((X.shape[0], X.shape[1]))    # (307, 2)
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])

# 我们还需要为验证集（使用相同的模型参数）执行此操作。
# 我们将使用与真实标签组合的这些概率来确定将数据点分配为异常的最佳概率阈值。
pval = np.zeros((Xval.shape[0], Xval.shape[1]))   # (307, 2)
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

# 接下来，我们需要一个函数，找到给定概率密度值和真实标签的最佳阈值。
# 为了做到这一点，我们将为不同的epsilon值计算F1分数。
#  F1是真阳性，假阳性和假阴性的数量的函数。 方程式在练习文本中。
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon      # pred是true或者false   小于epsilon则标为异常

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


epsilon, f1 = select_threshold(pval, yval)

print("epsilon= ", epsilon)
print("f1= ", f1)

outliers = np.where(p < epsilon)   # 返回满足条件的下标

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])     # outliers[0]行下标
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()

