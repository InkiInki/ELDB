"""
作者: 因吉
联系方式: inki.yinji@qq.com
创建日期：2020 0922
近一次修改：2021 0713
说明：获取距离矩阵
"""


import os
import numpy as np
from Code.Function import print_progress_bar
# 由于需要进行文件读取，所有这里进行了存储精度的控制
np.set_printoptions(precision=6)


def ave_hausdorff(bag1, bag2):
    """
    平均Hausdorff距离，相关文献可以参考：
        "Multi-instance clustering with applications to multi-instance prediction."
    :param
        bag1:   数据包1，需要使用numpy格式，形状为$n1 \times d$，其中$n1$为包的大小，$d$为实例的维度
        bag2：   类似于包1
    :return
        两个包的距离度量
    """
    # 统计总距离值
    sum_dis = 0
    for ins1 in bag1:
        # 计算当前实例与最近实例的距离
        temp_min = np.inf
        for ins2 in bag2:
            temp_min = min(i2i_euclidean(ins1, ins2), temp_min)
        sum_dis += temp_min

    for ins2 in bag2:
        temp_min = np.inf
        for ins1 in bag1:
            temp_min = min(i2i_euclidean(ins2, ins1), temp_min)
        sum_dis += temp_min

    return sum_dis / (len(bag1) + len(bag2))


def simple_dis(bag1, bag2):
    """
    相关参数请参照平均Hausdorff距离
    说明：
        使用两个包均值向量之间的欧式距离来代替包之间的距离
    """

    return i2i_euclidean(np.average(bag1, 0), np.average(bag2, 0))


def i2i_euclidean(ins1, ins2):
    """
    欧式距离
    :param
        ins1：  向量1，为numpy类型，且$\in \mathcal{R}^d$
        ins2：  向量2
    @return
        两个向量的欧式距离值
    """
    return np.sqrt(np.sum((ins1 - ins2)**2))


class B2B:
    """
    用于初始化数据集相关的包距离矩阵
    :param
        data_name：      数据集名称，用于存储文件的命名
        bags：           整个包空间，格式详见musk1+等数据集
        b2b_type：       包之间距离度量的方式，已有的包括：平均Hausdorff ("ave")距离和simple_dis ("sim)
        b2b_save_home：  默认距离矩阵的存储主目录
    """

    def __init__(self, data_name, bags, b2b_type="ave", b2b_save_home="../Data/Distance/b2b_"):
        """
        构造函数
        """
        # 传递的参数
        self._data_name = data_name
        self._bags = bags
        self._b2b_type = b2b_type
        self._b2b_save_home = b2b_save_home
        self.__initialize__b2b()

    def __initialize__b2b(self):
        """
        初始化函数
        """
        # 存储计算的距离矩阵
        self._dis = []
        # 获取距离矩阵的存储路径
        self._save_b2b_path = self._b2b_save_home + self._data_name + '_' + self._b2b_type + ".npz"
        self.__compute_dis()

    def __compute_dis(self):
        """
        计算距离
        """
        if not os.path.exists(self._save_b2b_path):
            # 包的大小
            N = len(self._bags)
            dis = np.zeros((N, N))
            print("计算距离矩阵...")
            for i in range(N):
                # 打印进度条
                print_progress_bar(i, N)
                # 包i和j的距离即j和i的距离
                for j in range(i, N):
                    if self._b2b_type == 'ave':
                        dis[i, j] = dis[j, i] = ave_hausdorff(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
                    else:
                        dis[i, j] = dis[j, i] = simple_dis(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
            # 结束的时候需要换行一下
            print()
            np.savez(self._save_b2b_path, dis=dis)
        self.dis = np.load(self._save_b2b_path)['dis']

    def get_dis(self):
        """
        获取距离矩阵
        """
        return self._dis


def test():
    """
    测试函数
    """
    data_name = "../Data/Benchmark/musk1+.mat"
    from Gui.MIL import MIL
    a = MIL(data_name)
    b = B2B("musk1+", a.bags)


if __name__ == '__main__':
    test()
