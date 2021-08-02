"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建日期：2021 0713
近一次修改：2021 0714
说明：一些常用的函数
"""


import numpy as np
from scipy.io import loadmat


def print_progress_bar(idx, size):
    """
    打印进度条
    :param
        idx:    当前位置
        size：   总进度
    """
    print('\r' + '▇' * int(idx // (size / 50)) + str(np.ceil((idx + 1) * 100 / size)) + '%', end='')


def load_file(data_path):
    """
    载入.mat类型的多示例数据集
    :param
        data_path:  数据集的存储路径
    """
    return loadmat(data_path)['data']
