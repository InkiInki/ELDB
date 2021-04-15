"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0922; last modified in 2020 1020
@note: Distance or similarity function between instance and bag,
and all vector data's type must be numpy.array.
"""

from Gui.B2B import *


def max_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the similarity between bag and discriminative instance.
    :param
        para_bag:
            The given bag, and its have not instance label.
        para_ins:
            The given discriminative instance.
        para_ins_dis:
            The type of distance / similarity function for two instances.
                1. 'euclidean': the euclidean distance.
                2. 'rbf': the RBF kernel.
            And its default setting is 'rbf'.
        para_gamma:
            The gamma for RBF function.
    :return
        The maximum rbf value.
    """
    ret_dis = -np.inf

    if para_ins_dis == 'rbf':
        for ins in para_bag:
            ret_dis = max(ret_dis, kernel_rbf(ins, para_ins, para_gamma))
    elif para_ins_dis == 'euclidean':
        ret_dis = np.inf
        for ins in para_bag:
            ret_dis = min(ret_dis, dis_euclidean(ins, para_ins))
    elif para_ins_dis == 'rbf2':
        for ins in para_bag:
            ret_dis = max(ret_dis, np.exp(-para_gamma * (dis_euclidean(ins, para_ins)**2)))

    if ret_dis == -1:
        print("Fetal error: the similarity between bag and instance is -1.")

    return ret_dis


def ave_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the average similarity.
    More detail please refer max_similarity.
    """
    ret_dis = 0

    if para_ins_dis == 'rbf':
        for ins in para_bag:
            ret_dis += kernel_rbf(ins, para_ins, para_gamma)
    elif para_ins_dis == 'euclidean':
        for ins in para_bag:
            ret_dis += dis_euclidean(ins, para_ins)
    elif para_ins_dis == 'rbf2':
        for ins in para_bag:
            ret_dis += np.exp(-para_gamma * (dis_euclidean(ins, para_ins)**2))

    if ret_dis == -1:
        print("Fetal error: the similarity between bag and instance is -1.")

    return ret_dis / len(para_bag)


def ave_h_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the average similarity, and the equation is refer to the average Hausdorff distance.
    If you want learn more, please read this paper:
    "Multi-instance clustering with applications to multi-instance prediction".
    """
    return ave_hausdorff(para_bag, np.resize(para_ins, (1, len(para_ins))), para_ins_dis, para_gamma)


def min_h_similarity(para_bag, para_ins, para_ins_dis='rbf', para_gamma=1):
    """
    Compute the minimum similarity, and the equation is refer to the maximum Hausdorff distance.
    More learn, please refer the ave_similarity.
    """
    return max_hausdorff(para_bag, np.resize(para_ins, (1, len(para_ins))), para_ins_dis, para_gamma)


def get_i2b_sim(para_bag, para_ins, para_i2b_dis='max', para_i2i_dis='rbf2', para_gamma=1):
    """
    Get the distance or similarity matrix by employing the given distance or similarity function for i2b.
    :param
        para_i2b_dis:
            The given i2b distance or similarity function type.
        others:
            Please refer the max_similarity function.
    :return
        The distance or similarity matrix.
    """
    if para_i2b_dis == 'max':
        return max_similarity(para_bag, para_ins, para_i2i_dis, para_gamma)


class I2B:
    """
    The class of i2b.
    :param
        para_i2b_type:
            The distance or similarity function type for instances and bags.
        others:
            Please refer the Distance or I2I.
    """

    def __init__(self, para_name, para_bags, para_dimension,
                 para_i2b_type='max',
                 para_i2b_i2i_type='euclidean',
                 para_gamma=1):
        """
        The constructor.
        """
        self.name = para_name
        self.bags = para_bags
        self.dimension = para_dimension
        self.i2b_type = para_i2b_type
        self.i2b_i2i_type = para_i2b_i2i_type
        self.gamma = para_gamma
        self.i2b_dis = []
        self.i2b_dis_path = ''

    def __initialize_i2b(self):
        """
        The initialize of i2b.
        """
        self.dis_path = '../Data/Distance/i2b_' + \
                        self.name + '_' + self.i2b_type + '_' + self.i2b_i2i_type
        if self.i2b_i2i_type == 'rbf' or self.i2b_i2i_type == 'rbf2':
            self.dis_path += '_' + str(self.gamma)
        self.dis_path += '.npz'
