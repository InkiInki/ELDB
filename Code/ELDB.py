"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建日期: 2020 1029
进一次修改:2021 0719
"""

import numpy as np
import warnings
from Code.ClassifyTool import Classify
from Code.Distance import B2B
from Code.Function import get_k_cv_idx, get_iter, get_performance
from Code.MIL import MIL
warnings.filterwarnings('ignore')


class ELDB(MIL):
    """
    ELDB算法主类
    """

    def __init__(self, data_path, psi=0.9, alpha=0.75, batch=None, psi_max=100,
                 type_b2b="ave", mode_bag_init="g", mode_action="a", k=10,
                 type_classifier=None, type_performance=None, print_loop=False,
                 save_home="../Data/Distance/", bag_space=None):
        """
        构造函数
        :param
            data_path:              数据的存储路径
            psi:                    基础dBagSet的大小
            alpha:                  学习率，即基础dBagSet的大小与训练集的比值
                                     在算法中，$\alpha \times N$ 表示为T_d，余下作为$T_s$
            batch:                  批次大小，当指定为None为，将使用默认划分，将$T_s$二分
            psi_max:                基础dBagSet的最大容量
            type_b2b:               距离函数的类型
            mode_bag_init:          基础dBagSet的初始化模式
            mode_action:            算法的行为模式
            k:                      k折交叉验证
            type_classifier:        单实例分类器，默认None时将使用["knn", "svm", "j48"]
            type_performance:       性能度量类型，默认None时将使用["acc", "f1_score"]
            print_loop:             是否输出每一折的轮次
            save_home:
            bag_space:              参见MIL文件
        """
        super(ELDB, self).__init__(data_path, save_home=save_home, bag_space=bag_space)
        self._psi = psi
        self._alpha = alpha
        self._batch = batch
        self._psi_max = psi_max
        self._type_b2b = type_b2b
        self._mode_bag_init = mode_bag_init
        self._mode_action = mode_action
        self._k = k
        self._type_classifier = type_classifier
        self._type_performance = type_performance
        self._print_loop = print_loop
        self.__init_eldb()

    def __init_eldb(self):
        """
        ELDB的初始化函数
        """
        self._type_classifier = ["knn", "svm", "j48"] if self._type_classifier is None else self._type_classifier
        self._type_performance = ["accuracy", "f1_score"] if self._type_performance is None else self._type_performance
        # 距离矩阵
        self.dis = B2B(self.data_name, self.bag_space, self._type_b2b, self.save_home).get_dis()
        # 记录不同分类器不同分类性能的分类结果
        self.lab_predict = {}
        # 记录按照交叉验证顺利的真实标签
        self.lab_true = []
        # 记录分类性能
        self.val_performance = {}

    def __reset_record(self):
        """
        重设记录相关的变量
        """
        self.lab_predict = {}
        self.lab_true = []
        self.val_performance = {}

    def __get_classifier(self):
        """
        获取分类器
        """
        return Classify(self._type_classifier, self._type_performance)

    def get_state(self):
        """
        获取使用的分类以及度量性能
        """
        return self._type_classifier, self._type_performance

    def get_mapping(self):
        """
        获取映射结果.
        """

        def __dBagSet_update_r(para_idx_dBagSet, para_score_dBagSet, idx_cur, score_cur):
            """
            用于行为模式r的更新
            :param
                para_score_td:   T_d的得分拷贝
                score_cur:       当前包的得分
            :return
                返回更新后的score_td
            """
            for idx_find in np.arange(len(para_idx_dBagSet) - 1, -1, -1):
                if score_cur > para_score_dBagSet[idx_find]:
                    continue
                else:
                    idx_find += 1
                    idx_find = len(para_idx_dBagSet) - 1 if idx_find == len(para_idx_dBagSet) else idx_find
                    para_idx_dBagSet[idx_find + 1:] = para_idx_dBagSet[idx_find: -1]
                    para_score_dBagSet[idx_find + 1:] = para_score_dBagSet[idx_find: -1]
                    para_idx_dBagSet[idx_find], para_score_dBagSet[idx_find] = idx_cur, score_cur
                    break
            return para_idx_dBagSet, para_score_dBagSet

        # 获取训练集和测试集的索引
        idxes_tr, idxes_te = get_k_cv_idx(self.N, self._k)
        # 获取单实例分类器
        classifier = self.__get_classifier()
        # 性能度量器
        performance = get_performance(self._type_performance)
        # 记录参数重设
        self.__reset_record()
        # 主循环
        for loop, (idx_tr, idx_te) in enumerate(zip(idxes_tr, idxes_te)):
            """步骤0:初始化操作"""
            if self._print_loop:
                print("第{}折交叉验证...".format(loop))
            # 计算训练集、基准数据集和更新数据集的大小
            N_T = len(idx_tr)
            N_Ts = int(N_T * (1 - self._alpha))
            # 计算批次大小
            batch = N_Ts // 2 if self._batch is None else self._batch
            # 计算最大更新次数
            n_l = N_Ts // batch
            N_Td = N_T - (n_l * batch)
            # 获取T_d和T_s的索引
            idx_td, idx_ts = np.array(idx_tr[:N_Td]), np.array(idx_tr[N_Td:])

            """步骤1:模型和参数初始化"""
            # 计算\Delta矩阵
            matrix_Delta = np.zeros((N_Td, N_Td), dtype=int)
            for i in range(N_Td):
                for j in range(N_Td):
                    # 这里使用最简单的设计，即标签相同设置为-1；反之为1
                    if self.bag_lab[idx_td[i]] == self.bag_lab[idx_td[j]]:
                        matrix_Delta[i, j] = -1
                    else:
                        matrix_Delta[i, j] = 1
            # 计算\Gamma矩阵
            matrix_Gamma = np.diag(np.sum(matrix_Delta, 1))
            # 计算L矩阵
            matrix_L = matrix_Gamma - matrix_Delta
            # 只需要保留L矩阵
            del matrix_Delta, matrix_Gamma
            # 基于整个T_d进行映射
            mapping_bag = self.dis[idx_tr, :][:, idx_td]
            # 使用矩阵乘法加速
            score_t = np.dot(np.dot(mapping_bag, matrix_L), np.transpose(mapping_bag))
            # 对角元素即是包的得分
            score_t = np.diag(score_t)
            # 获取T_d和T_s中每一个包的得分
            score_td, score_ts = score_t[:N_Td], score_t[N_Td:]
            # 获取初始dBagSet的大小
            psi = int(min(self._psi_max, N_Td) * self._psi)
            arg_score_td = np.argsort(score_td)[::-1]
            # 获取dBagSet在训练集中的真实索引
            idx_dBagSet = arg_score_td[:psi].tolist()
            score_dBagSet, idx_dBagSet = score_td[idx_dBagSet], [idx_td[idx_dBagSet].tolist()]
            del score_t, arg_score_td
            # 记录最小得分的索引和得分
            tau, p = len(idx_dBagSet[-1]) - 1, score_dBagSet[-1]
            # 获取所有更新后的dBagSet
            for i in range(n_l):
                idx_dBagSet_update, score_dBagSet_update = idx_dBagSet[-1].copy(), score_dBagSet.copy()
                for j in range(batch):
                    # 分数小于等于的均不考虑；取等是因为两个包得分完全相等的概率很小
                    idx_temp = i * batch + j
                    if score_ts[idx_temp] <= p:
                        continue
                    # 行为”a“只需要添加即可
                    if self._mode_action == "a":
                        idx_dBagSet_update.append(idx_ts[idx_temp])
                    # 行为”r“需要不断替换操作
                    else:
                        idx_dBagSet_update, score_dBagSet = __dBagSet_update_r(
                            idx_dBagSet_update, score_dBagSet_update, idx_ts[idx_temp], score_ts[idx_temp])

                if idx_dBagSet_update == idx_dBagSet[-1]:
                    continue
                idx_dBagSet.append(idx_dBagSet_update)
            del idx_dBagSet_update, score_dBagSet, mapping_bag

            """步骤2:构建带权集成模型"""
            # 遍历每一个dBagSet
            Y_d, Y_s = self.bag_lab[idx_td], self.bag_lab[idx_ts]
            # 训练集和测试集标签
            lab_tr, lab_te = self.bag_lab[idx_tr], self.bag_lab[idx_te]
            Predict, Weight = [], []

            for i, dBagSet in enumerate(idx_dBagSet):
                # 获取映射
                mapping_td, mapping_ts = self.dis[idx_td, :][:, dBagSet], self.dis[idx_ts, :][:, dBagSet]
                # 获取迭代器
                data_iter = get_iter(mapping_td, Y_d, mapping_ts, Y_s)
                # 获取权重并记录
                Weight.append(classifier.test(data_iter))
                # 获取训练集和测试集
                mapping_tr, mapping_te = np.vstack([mapping_td, mapping_ts]), self.dis[idx_te, :][:, dBagSet]
                del mapping_td, mapping_ts
                # 模型重训练并预测
                data_iter = get_iter(mapping_tr, lab_tr, mapping_te, lab_te)
                classifier.test(data_iter)
                Predict.append(classifier.te_predict_arr)
            # 清理缓存
            del data_iter, mapping_tr, mapping_te, lab_tr, lab_te
            """步骤3:获取预测的标签向量"""
            # 记录预测结果
            te_predict_all = {}
            # 处理每一个分类器和分类指标的预测结果
            for i, (predict, weight) in enumerate(zip(Predict, Weight)):
                # 对于每一个分类器
                for classifier_name in self._type_classifier:
                    # 对于每一个分类度量指标
                    weight_classifier = weight[classifier_name]
                    for j, metric in enumerate(weight_classifier):
                        # 加权
                        weight_predict = metric * np.array(predict[classifier_name])
                        te_predict_all[classifier_name + ' ' + self._type_performance[j]] = \
                            te_predict_all.get(classifier_name + ' ' + self._type_performance[j],
                                               np.zeros_like(weight_predict)) + weight_predict
            # 计算分类阈值
            tau = len(Weight) / 2

            # 阈值函数
            def sign(arr):
                arr[arr > tau] = 1
                arr[arr != 1] = 0
                return arr
            for key, val in te_predict_all.items():
                if loop == 0:
                    self.lab_predict[key] = sign(val).tolist()
                else:
                    self.lab_predict[key].extend(sign(val).tolist())
            self.lab_true.extend(self.bag_lab[idx_te])
        # 获取分类性能
        for key, val in self.lab_predict.items():
            key_temp = key.split()
            self.val_performance[key] = performance[key_temp[-1]](val, self.lab_true)

        return self.val_performance
