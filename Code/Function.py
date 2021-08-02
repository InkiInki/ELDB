"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建日期:2021 0713
近一次修改:2021 0714
说明:一些常用的函数
"""


import numpy as np
from scipy.io import loadmat


def get_iter(tr, tr_lab, te, te_lab):
    """
    获取单词迭代器
    :param  tr:     训练集
    :param  tr_lab: 训练集标签
    :param  te:     测试集
    :param  te_lab: 测试集标签
    :return 相应迭代器
    """
    yield tr, tr_lab, te, te_lab


def get_k_cv_idx(num_x, k=10):
    """
    获取k次交叉验证的索引
    :param num_x:       数据集的大小
    :param k:           决定使用多少折的交叉验证
    :return:            训练集索引，测试集索引
    """
    # 随机初始化索引
    rand_idx = np.random.permutation(num_x)
    # 每一折的大小
    fold = int(np.floor(num_x / k))
    ret_tr_idx = []
    ret_te_idx = []
    for i in range(k):
        # 获取当前折的训练集索引
        tr_idx = rand_idx[0: i * fold].tolist()
        tr_idx.extend(rand_idx[(i + 1) * fold:])
        ret_tr_idx.append(tr_idx)
        # 添加当前折的测试集索引
        ret_te_idx.append(rand_idx[i * fold: (i + 1) * fold].tolist())
    return ret_tr_idx, ret_te_idx


def print_progress_bar(idx, size):
    """
    打印进度条
    :param idx:    当前位置
    :param size:   总进度
    """
    print('\r' + '▇' * int(idx // (size / 50)) + str(np.ceil((idx + 1) * 100 / size)) + '%', end='')


def load_file(data_path):
    """
    载入.mat类型的多示例数据集
    :param data_path:  数据集的存储路径
    """
    return loadmat(data_path)['data']
