"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0922; last modified in 2020 1021.
@note: Distance or similarity function for multi-instance learning (MIL),
and all vector data's type must be numpy.array.
"""


import os
from Gui.I2I import *
from Gui.SimpleTool import *
np.set_printoptions(precision=6)


def ave_hausdorff(para_mat1, para_mat2, para_ins_dis='rbf', para_gamma=1):
    """
    The ave-Hausdorff distance measure for MIL, and its reference is the Eq. (7) from the article named
        "Multi-instance clustering with applications to multi-instance prediction."
    @param:
        para_mat1:
            The given mat, e.g., np.array([[1, 2], [0, 1], [0, 1.1]]).
        para_mat2:
            The given mat like para_mat1, e.g., np.array([[1, 2.1], [0, 0.9], [0, 1.0]])
        para_ins_dis:
            The distance measure for SIL, and its default setting is 'rbf'.
        para_gamma:
            The gamma for rbf function, and its defaulting setting is 1.
        para_owa_type:
            The type of OWA, and its including 'linear_decrease', 'inverse_additive'.
    @return:
        A scalar for two matrix.
        Specially, the return is a scalar for two bags for MIL.
    @note:
        The two given matrix must have the same dimensions.
    """
    temp_len_mat1 = len(para_mat1)
    temp_len_mat2 = len(para_mat2)

    temp_sum = 0
    if para_ins_dis == 'rbf':
        for i in range(temp_len_mat1):
            temp_min = -np.inf
            for j in range(temp_len_mat2):
                temp_dis = kernel_rbf(para_mat1[i], para_mat2[j], para_gamma)
                temp_min = max(temp_dis, temp_min)
            temp_sum += temp_min

        for j in range(temp_len_mat2):
            temp_min = -np.inf
            for i in range(temp_len_mat1):
                temp_dis = kernel_rbf(para_mat2[j], para_mat1[i])
                temp_min = max(temp_dis, temp_min)
            temp_sum += temp_min
    elif para_ins_dis == 'euclidean':
        for i in range(temp_len_mat1):
            temp_min = np.inf
            for j in range(temp_len_mat2):
                temp_dis = dis_euclidean(para_mat1[i], para_mat2[j])
                temp_min = min(temp_dis, temp_min)
            temp_sum += temp_min

        for j in range(temp_len_mat2):
            temp_min = np.inf
            for i in range(temp_len_mat1):
                temp_dis = dis_euclidean(para_mat2[j], para_mat1[i])
                temp_min = min(temp_dis, temp_min)
            temp_sum += temp_min
    elif para_ins_dis == 'rbf2':
        for i in range(temp_len_mat1):
            temp_min = -np.inf
            for j in range(temp_len_mat2):
                temp_dis = np.exp(-para_gamma * (dis_euclidean(para_mat1[i], para_mat2[j])**2))
                temp_min = max(temp_dis, temp_min)
            temp_sum += temp_min
        for j in range(temp_len_mat2):
            temp_min = -np.inf
            for i in range(temp_len_mat1):
                temp_dis = np.exp(-para_gamma * (dis_euclidean(para_mat2[j], para_mat1[i])**2))
                temp_min = max(temp_dis, temp_min)
            temp_sum += temp_min

    return temp_sum / (temp_len_mat1 + temp_len_mat2)


def max_hausdorff(para_mat1, para_mat2, para_ins_dis='rbf', para_gamma=1):
    """
    The max-Hausdorff distance measure for MIL, and its reference is the article named
        "Measure, topology, and fractal geometry, 3rd print."
    @note:
        Please refer the ave_hausdorff.
    """
    temp_len_para_mat1 = len(para_mat1)
    temp_len_para_mat2 = len(para_mat2)

    temp_max1 = -1
    temp_max2 = -1
    if para_ins_dis == 'rbf':
        for i in range(temp_len_para_mat1):
            temp_min = -np.inf
            for j in range(temp_len_para_mat2):
                temp_dis = kernel_rbf(para_mat1[i], para_mat2[j], para_gamma)
                temp_min = max(temp_dis, temp_min)
            temp_max1 = min(temp_min, temp_max1)

        for j in range(temp_len_para_mat2):
            temp_min = -np.inf
            for i in range(temp_len_para_mat1):
                temp_dis = kernel_rbf(para_mat1[j], para_mat2[i], para_gamma)
                temp_min = max(temp_dis, temp_min)
            temp_max2 = min(temp_min, temp_max2)
    elif para_ins_dis == 'euclidean':
        for i in range(temp_len_para_mat1):
            temp_min = np.inf
            for j in range(temp_len_para_mat2):
                temp_dis = dis_euclidean(para_mat1[i], para_mat2[j])
                temp_min = min(temp_dis, temp_min)
            temp_max1 = max(temp_min, temp_max1)

        temp_max2 = np.zeros(temp_len_para_mat2)
        for j in range(temp_len_para_mat2):
            temp_min = np.inf
            for i in range(temp_len_para_mat1):
                temp_dis = dis_euclidean(para_mat2[j], para_mat1[i])
                temp_min = min(temp_dis, temp_min)
            temp_max2 = max(temp_min, temp_max2)

    return max(temp_max1, temp_max2)


def min_hausdorff(para_mat1, para_mat2, para_ins_dis='rbf', para_gamma=1):
    """
    The min-Hausdorff distance measure for MIL, and its reference is the article named
        "Solving multiple-instance problem: a lazy learning approach."
    @note:
        Please refer the ave_hausdorff.

    """
    temp_len_mat1 = len(para_mat1)
    temp_len_mat2 = len(para_mat2)
    temp_sum_len = temp_len_mat1 * temp_len_mat2

    k = 0
    ret_min = 0
    if para_ins_dis == 'rbf':
        ret_min = -np.inf
        for i in range(temp_len_mat1):
            for j in range(temp_len_mat2):
                ret_min = max(ret_min, kernel_rbf(para_mat1[i], para_mat2[j], para_gamma))
                k += 1
    elif para_ins_dis == 'euclidean':
        ret_min = np.inf
        for i in range(temp_len_mat1):
            for j in range(temp_len_mat2):
                ret_min = min(ret_min, dis_euclidean(para_mat1[i], para_mat2[j]))
                k += 1

    return ret_min


def vir_hausdorff(para_mat1, para_mat2, para_ins_dis='rbf', para_gamma=1):
    """
    The virtual-Hausdorff distance measure for MIL
    @note:
        Please refer the ave_hausdorff.
    """
    temp_ins1 = np.average(para_mat1, 0)
    temp_ins2 = np.average(para_mat2, 0)

    if para_ins_dis == 'rbf':
        return kernel_rbf(temp_ins1, temp_ins2, para_gamma)
    elif para_ins_dis == 'euclidean':
        return dis_euclidean(temp_ins1, temp_ins2)


class B2B:
    """
    The class of get the distance or similarity between bags.
    :param
        para_name:
            The data set name.
        para_bags:
            The given bags, and its structure is $\{ (X_1, y_1), \cdots, (X_n, y_n) \}$.
            And we need the $X_i$.
        para_dis:
            The distance or similarity function for bags.
        para_ins_dis:
            The distance or similarity function for instances.
        para_gamma:
            The gamma for rbf kernel.
    @attribute
        dis:
            The distance or similarity matrix.
    @example
        # >>> b2b = Distance('musk1', para_bags)
        # >>> b2b.dis
    """

    def __init__(self, para_name, para_bags, para_dimension,
                 para_dis_type='ave_hausdorff',
                 para_ins_dis_type='euclidean',
                 para_gamma=1):
        """
        The constructor.
        """
        self.name = para_name
        self.bags = para_bags
        self.dimension = para_dimension
        self.dis_type = para_dis_type
        self.ins_dis_type = para_ins_dis_type
        self.gamma = para_gamma
        self.dis = []
        self.dis_path = ''
        self.__initialize__b2b()
        self.__get_dis()

    def __initialize__b2b(self):
        """
        The initialize for b2b.
        """
        self.dis_path = '../Data/Distance/b2b_' + self.name + '_' + self.dis_type + '_' + self.ins_dis_type
        if self.ins_dis_type == 'rbf' or self.ins_dis_type == 'rbf2':
            self.dis_path += '_' + str(self.gamma)
        self.dis_path += '.npz'

    def __get_dis(self):
        """
        Get the distance or similarity matrix.
        """
        if not os.path.exists(self.dis_path) or os.path.getsize(self.dis_path) == 0:
            np.savez(self.dis_path, dis=None)

            temp_size = len(self.bags)
            temp_dis = np.zeros((temp_size, temp_size))
            print("Computing the distance matrix...")
            for i in range(temp_size):
                print_progress_bar(i, temp_size)
                for j in range(i, temp_size):
                    if self.dis_type == 'ave_hausdorff':
                        temp_dis[i, j] = temp_dis[j, i] = ave_hausdorff(self.bags[i, 0][:, : self.dimension],
                                                                        self.bags[j, 0][:, : self.dimension],
                                                                        self.ins_dis_type,
                                                                        self.gamma)
                    elif self.dis_type == 'vir_hausdorff':
                        temp_dis[i, j] = temp_dis[j, i] = vir_hausdorff(self.bags[i, 0][:, : self.dimension],
                                                                        self.bags[j, 0][:, : self.dimension],
                                                                        self.ins_dis_type,
                                                                        self.gamma)
            print()
            np.savez(self.dis_path, dis=temp_dis)
        self.dis = np.load(self.dis_path)['dis']

    def get_dis(self):
        """
        Get the distance matrix.
        :return
            The distance matrix.
        """
        return self.dis
