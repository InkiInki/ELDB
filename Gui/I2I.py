"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0703; last modified in 2020 1231.
@note: Distance or similarity function for single-instance learning (SIL),
and all vector data's type must be numpy.array.
"""

import os
import numpy as np
np.set_printoptions(precision=6)

__all__ = ['dis_euclidean',
           'kernel_gaussian',
           'kernel_rbf']


def dis_euclidean(para_arr1, para_arr2):
    """The eucildean distance, i.e.m $||para_arr1 - para_arr2||^2$
    @param:
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
    @return
        A scalar.
    """
    return np.sqrt(np.sum((para_arr1 - para_arr2)**2))


def kernel_gaussian(para_arr1, para_arr2, para_gamma=1):
    """
    The details please refer the kernel_rbf.
    """
    return np.exp(-para_gamma * dis_euclidean(para_arr1, para_arr2)**2)
    

def kernel_rbf(para_arr1, para_arr2, para_gamma=1):
    r"""
    The Gaussian RBF kernel for SIL, i.e., $exp (\gama ||para_arr1 - para_arr2||^2)$.
    @param: 
    ------------
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
        para_gama:
            The gama for RBF kernel.
    ------------
    @return:
    ------------
        A scalar.
    ------------
    """
    
    return np.exp(-para_gamma * dis_euclidean(para_arr1, para_arr2))


def kernel_isk(arr1, arr2, forest):
    """
    Compute the similarity between two arrays by using the isolation kernel.
    """
    temp_num_forest = len(forest)
    temp_count = 0
    for i in range(temp_num_forest):
        if kernel_isk_tree(arr1, arr2, forest[i]):
            temp_count += 1

    return temp_count / temp_num_forest


def kernel_isk_tree(arr1, arr2, tree):
    """
    The flag function \mathbb{I}.
    """
    if tree is not None:
        temp_value = tree.value

        if len(temp_value) == 3:
            return True
        temp_attribute_idx, temp_threshold = temp_value[2:4]
        if arr1[temp_attribute_idx] < temp_threshold <= arr2[temp_attribute_idx]:
            return False
        if arr1[temp_attribute_idx] >= temp_threshold > arr2[temp_attribute_idx]:
            return False
        if tree.left is not None and \
                arr1[temp_attribute_idx] < temp_threshold and arr2[temp_attribute_idx] < temp_threshold:
            return kernel_isk_tree(arr1, arr2, tree.left)
        if tree.right is not None and \
                arr1[temp_attribute_idx] >= temp_threshold and arr2[temp_attribute_idx] >= temp_threshold:
            return kernel_isk_tree(arr1, arr2, tree.right)
    return True


class I2I:
    """
    The class of I2I.
    :param
        para_name, para_dimension, para_ins_dis_type, para_gamma
            The more detail please see the BIB.BIB.
        para_ins:
            The instance space.
    @attribute
        ins_dis:
            The distance or similarity matrix for instances.
    @example
        # >>> i2i = I2I('iris', para_ins, 166)
        # >>> i2i.ins_dis
    """

    def __init__(self, para_name, para_ins, para_dimension,
                 para_i2i_type='euclidean',
                 para_gamma=1,
                 para_is_save=True):
        """
        The constructor.
        """
        self.name = para_name
        self.ins = para_ins
        self.dimension = para_dimension
        self.i2i_type = para_i2i_type
        self.gamma = para_gamma
        self.is_save = para_is_save
        self.ins_dis_path = ''
        self.ins_dis = []
        self.__initialize_i2i()
        self.__get_ins_dis()
        
    def __initialize_i2i(self):
        """
        The initialize of i2i.
        """
        if self.is_save:
            self.ins_dis_path = '../Data/Distance/i2i_' + self.name + '_' + self.i2i_type
            if self.i2i_type == 'rbf' or self.i2i_type == 'rbf2':
                self.ins_dis_path += '_' + str(self.gamma)
            self.ins_dis_path += '.npz'

    def __get_ins_dis(self):
        """
        Get the distance or similarity matrix between instances.
        """
        if self.is_save:
            if not os.path.exists(self.ins_dis_path):
                np.savez(self.ins_dis_path, dis=None)
                self.ins_dis = self.__dis_choose()

            np.savez(self.ins_dis_path, dis=self.ins_dis)
            self.ins_dis = np.load(self.ins_dis_path)['dis']
        else:
            self.ins_dis = self.__dis_choose()

    def __dis_choose(self):
        """
        Using the choosing distance or similarity function.
        :return
            The distance or similarity matrix.
        """
        temp_num_ins = len(self.ins)
        temp_dis = np.zeros((temp_num_ins, temp_num_ins))
        for i in range(temp_num_ins):
            for j in range(i, temp_num_ins):
                if self.i2i_type == 'euclidean':
                    temp_dis[i, j] = temp_dis[j, i] = dis_euclidean(self.ins[i][: self.dimension],
                                                                    self.ins[j][: self.dimension])
        return temp_dis

    def get_dis(self):
        """
        Get the distance or similarity matrix.
        :return
            The distance or similarity matrix.
        """
        return self.ins_dis


class I2IBag:
    """
    The I2I class for MIL bags.
    :param
        Please refer the the class named Distance.Distance.
    @example
        # >>> i2i_bag = I2IBag('musk1', para_bags, 166)
        # >>> dis = i2i_bag.get_dis()
    """

    def __init__(self, para_name, para_bags, para_dimension,
                 para_i2i_type='euclidean',
                 para_gamma=1):
        """
        The constructor.
        """
        self.name = para_name
        self.bags = para_bags
        self.num_bags = 0
        self.dimension = para_dimension
        self.i2i_type = para_i2i_type
        self.gamma = para_gamma
        self.bag_ins_dis = {}
        self.bag_ins_dis_path = ''
        self.__initialize_bag_i2i()

    def __initialize_bag_i2i(self):
        """
        The initialize for b2b.
        """
        self.bag_ins_dis_path = '../Data/Distance/i2i_bag_' + self.name + '_' + self.i2i_type
        if self.i2i_type == 'rbf' or self.i2i_type == 'rbf2':
            self.bag_ins_dis_path += '_' + str(self.gamma)
        self.bag_ins_dis_path += '.npz'

        self.num_bags = len(self.bags)
        if not os.path.exists(self.bag_ins_dis_path) or os.path.getsize(self.bag_ins_dis_path) == 0:
            open(self.bag_ins_dis_path, 'a').close()
            self.__get_dis()
            np.savez(self.bag_ins_dis_path, bag_ins_dis=self.bag_ins_dis)
        self.bag_ins_dis = np.load(self.bag_ins_dis_path)['bag_ins_dis']

    def __get_dis(self):
        """
        Get the distance dict, and each value is a matrix for bag.
        """
        for i in range(self.num_bags):
            self.bag_ins_dis[i] = I2I(self.name, self.bags[i, 0][:, :self.dimension], self.dimension, self.i2i_type,
                                      self.gamma, False).get_dis()

    def get_dis(self):
        """
        Get the distance dict.
        """
        return self.bag_ins_dis.tolist()
