"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0903, last modified in 2020 1231.
@note: Some common function, and all given vector data's type must be numpy.array.
"""

import numpy as np
import scipy.io as scio


def get_iter(tr, tr_lab, te, te_lab):
    """
    Get iterator.
    :param
        tr:
            The training set.
        tr_lab:
            The training set's label.
        te:
            The test set.
        te_lab:
            The test set's label.
    """
    yield tr, tr_lab, te, te_lab


def is_print(para_str, para_is_print=True):
    """
    Is print?
    :param
        para_str:
            The print string.
        para_is_print:
            True print else not.
    """
    if para_is_print:
        print(para_str)


def load_file(para_path):
    """
    Load file.
    :param
        para_file_name:
            The path of the given file.
    :return
        The data.
    """
    temp_type = para_path.split('.')[-1]

    if temp_type == 'mat':
        ret_data = scio.loadmat(para_path)
        return ret_data['data']
    else:
        with open(para_path) as temp_fd:
            ret_data = temp_fd.readlines()

        return ret_data


def owa_weight(para_num, para_type='linear_decrease'):
    """
    The ordered weighted averaging operators (OWA) can replace the maximum or minimum operators.
    And the purpose of this function is to generate the owa weights.
    And the more refer is:
    R. R. Yager, J. Kacprzyk, The ordered weighted averaging operators: Theory and applications, Springer Science &
    Business Media, 2012.
    :param
        para_num:
            The length of weights list.
        para_type:
            'linear_decrease';
            'inverse_additive',
            and its default setting is 'linear_decrease'.
    :return
        The owa weights.
    """
    if para_num == 1:
        return np.array([1])
    else:
        if para_type == 'linear_decrease':
            temp_num = 2 / para_num / (para_num + 1)
            return np.array([(para_num - i) * temp_num for i in range(para_num)])
        elif para_type == 'inverse_additive':
            temp_num = np.sum([1 / i for i in range(1, para_num + 1)])
            return np.array([1 / i / temp_num for i in range(1, para_num + 1)])
        else:
            return owa_weight(para_num)


def print_go_round(para_idx, para_str='Program processing'):
    """
    Print the round.
    :param
        para_idx:
            The current index.
        para_str:
            The print words.
    """
    round_list = ["\\", "|", "/", "-"]
    print('\r' + para_str + ': ' + round_list[para_idx % 4], end="")


def print_progress_bar(para_idx, para_len):
    """
    Print the progress bar.
    :param
        para_idx:
            The current index.
        para_len:
            The loop length.
    """
    print('\r' + '▇' * int(para_idx // (para_len / 50)) + str(np.ceil((para_idx + 1) * 100 / para_len)) + '%', end='')


class Count(dict):
    """
    The count class with dict.
    """
    def __missing__(self, __key):
        return 0


class Tree:
    """
    A tree structure.
    """

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right
