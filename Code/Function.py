"""
作者: 因吉
联系方式: inki.yinji@qq.com
创建日期：2021 0713
近一次修改：2021 0714
说明：一些常用的函数
"""


import numpy as np


def print_progress_bar(idx, size):
    """
    打印进度条
    :param
        idx:    当前位置
        size：   总进度
    """
    print('\r' + '▇' * int(idx // (size / 50)) + str(np.ceil((idx + 1) * 100 / size)) + '%', end='')
