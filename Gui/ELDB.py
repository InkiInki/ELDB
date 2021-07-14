"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 1029, last modified in 2020 1205.
"""

import warnings
import numpy as np
from Gui.MIL import MIL
from Gui.Classify import Classify
from Gui.B2B import B2B
from Gui.SimpleTool import get_iter

warnings.filterwarnings('ignore')


class ELDB(MIL):
    """
    Multi-models RMDB.
    """

    def __init__(self, path, has_ins_label=True, k=10, learning_r=0.25, m=.1, num_batch=2, max_m=100,
                 update_mode='a', b2b_type=None, algorithm_type=None, classifier_type=None):
        """
        The constructor.
        """
        super(ELDB, self).__init__(path, has_ins_label)
        self.k = k
        self.learning_r = learning_r
        self.m = m
        self.num_batch = num_batch
        self.max_m = max_m
        self.update_mode = update_mode
        self.dis = {}
        self.tr_idx = []
        self.te_idx = []
        self.choose_mode_type = ['a', 'r']
        self.b2b_type = ['ave_hausdorff', 'vir_hausdorff'] if b2b_type is None else b2b_type
        self.algorithm_type = ['g', 'p', 'n', 'b'] if algorithm_type is None else algorithm_type
        self.classifier_type = ['knn', 'j48', 'svm'] if classifier_type is None else classifier_type
        self.__initialize_rmdb()

    def __initialize_rmdb(self):
        """
        The initialize of rmdb.
        """

        for b2b in self.b2b_type:
            self.dis[b2b] = B2B(self.data_name, self.bags, self.dimensions, b2b).get_dis()

    def __get_best_para(self, model, b2b, tr_idx, va_idx):
        """
        Include best distance function, algorithm type and classifier.
        :param
            model:
                The given model.
            b2b:
                The distance function.
            tr_idx:
                The training set index.
            va_idx:
                The validation set index.
        :return
            The best parameters.
        """
        ret_best_para = None
        temp_num_model = len(model)
        temp_weights = np.zeros((temp_num_model, len(self.classifier_type)))
        for i in range(temp_num_model):
            temp_tr = self.dis[b2b][tr_idx][:, model[i]]
            temp_tr_lab = self.bags_label[tr_idx]
            temp_va = self.dis[b2b][va_idx][:, model[i]]
            temp_va_lab = self.bags_label[va_idx]
            for j in range(len(self.classifier_type)):
                temp_weights[i, j] = Classify(get_iter(temp_tr, temp_tr_lab, temp_va, temp_va_lab), self.
                                              classifier_type[j]).get_result(para_type='f1-measure')
                if ret_best_para is None or temp_weights[i, j] > ret_best_para[0]:
                    ret_best_para = [temp_weights[i, j], temp_num_model, j, self.classifier_type[j]]

        ret_best_para.append(temp_weights[:, ret_best_para[2]].tolist())
        return ret_best_para

    def self_learning(self):
        """
        Get mapping.
        """

        self.tr_idx, self.te_idx = self.get_index(self.k)
        positive_label = np.max(self.bags_label)
        for loop in range(self.k):
            temp_num_tr = len(self.tr_idx[loop])
            temp_num_va = int(self.learning_r * temp_num_tr)
            temp_batch_size = int(temp_num_va // self.num_batch)
            temp_num_tr -= temp_num_va
            temp_tr_idx = self.tr_idx[loop][:temp_num_tr]
            temp_va_idx = self.tr_idx[loop][temp_num_tr:]

            # Step 1: Get Q.
            temp_q = np.zeros((temp_num_tr, temp_num_tr))
            for i in range(temp_num_tr):
                for j in range(temp_num_tr):
                    if self.bags_label[temp_tr_idx[i]] == self.bags_label[temp_tr_idx[j]]:
                        temp_q[i, j] = -1
                    else:
                        temp_q[i, j] = 1

            # Step 2: Get D.
            temp_d = np.sum(temp_q, 1)
            temp_d = np.diag(temp_d)

            # Step 3: Get L.
            temp_l = temp_d - temp_q
            del temp_q, temp_d

            # Step 4: Get initialization dbp.
            temp_score = {}
            temp_sorted_score_idx = {}
            for b2b in self.b2b_type:
                temp_score[b2b] = []
                for i in range(temp_num_tr):
                    temp_vec = self.dis[b2b][temp_tr_idx, :][:, temp_tr_idx[i]]
                    temp_score[b2b].append(np.dot(np.dot(temp_vec, temp_l), temp_vec))
                temp_sorted_score_idx[b2b] = np.argsort(temp_score[b2b]).tolist()[::-1]
                temp_score[b2b] = np.sort(temp_score[b2b]).tolist()[::-1]

            # Step 5: Initialization choose.
            temp_num_model = {}
            temp_model_set = {}
            temp_model_idx_set = {}
            for b2b in self.b2b_type:
                temp_num_model[b2b] = {}
                temp_model_set[b2b] = {}
                temp_model_idx_set[b2b] = {}
                for algorithm in self.algorithm_type:
                    temp_num_model[b2b][algorithm] = 0
                    temp_model_set[b2b][algorithm] = []
                    temp_model_idx_set[b2b][algorithm] = []
                    if algorithm == 'g':
                        temp_m = max(int(self.m * min(temp_num_tr, 100)), 1)
                        temp_model_set[b2b][algorithm].append(temp_score[b2b][:temp_m])
                        temp_model_idx_set[b2b][algorithm].append(temp_sorted_score_idx[b2b][:temp_m])
                    else:
                        temp_p_score = []
                        temp_n_score = []
                        temp_p_score_idx = []
                        temp_n_score_idx = []
                        for i in range(temp_num_tr):
                            if self.bags_label[temp_tr_idx[i]] == positive_label:
                                temp_p_score.append(temp_score[b2b][i])
                                temp_p_score_idx.append(temp_sorted_score_idx[b2b][i])
                            else:
                                temp_n_score.append(temp_score[b2b][i])
                                temp_n_score_idx.append(temp_sorted_score_idx[b2b][i])
                        temp_num_p = len(temp_p_score)
                        temp_num_n = len(temp_n_score)

                        temp_p_m = max(int(self.m * min(temp_num_p, 100)), 1)
                        temp_n_m = max(int(self.m * min(temp_num_n, 100)), 1)
                        if algorithm == 'p':
                            temp_model_set[b2b][algorithm].append(temp_p_score[:temp_p_m])
                            temp_model_idx_set[b2b][algorithm].append(temp_p_score_idx[:temp_p_m])
                        elif algorithm == 'n':
                            temp_model_set[b2b][algorithm].append(temp_n_score[:temp_n_m])
                            temp_model_idx_set[b2b][algorithm].append(temp_n_score_idx[:temp_n_m])
                        elif algorithm == 'b':
                            temp_m = min(max(int(self.m * min(temp_num_p, temp_num_n)), 1), 50)
                            temp_model_set[b2b][algorithm].append(temp_p_score[:temp_m] + temp_n_score[:temp_m])
                            temp_model_idx_set[b2b][algorithm]. \
                                append(temp_p_score_idx[:temp_m] + temp_n_score_idx[:temp_m])

            # Step 6: Update model and record.
            for batch in range(self.num_batch):
                temp_idx = temp_va_idx[batch * temp_batch_size: (batch + 1) * temp_batch_size]
                for b2b in self.b2b_type:
                    temp_batch_score = []
                    for idx in temp_idx:
                        temp_vec = self.dis[b2b][temp_tr_idx, :][:, idx]
                        temp_batch_score.append(np.dot(np.dot(temp_vec, temp_l), temp_vec))
                    for algorithm in self.algorithm_type:
                        temp_model = np.copy(temp_model_set[b2b][algorithm][temp_num_model[b2b][algorithm]])
                        temp_model_idx = np.copy(temp_model_idx_set[b2b][algorithm][temp_num_model[b2b][algorithm]])
                        temp_min_score_idx = np.argmin(temp_model)
                        temp_min_score = temp_model[temp_min_score_idx]
                        temp_model = temp_model.tolist()
                        temp_model_idx = temp_model_idx.tolist()
                        # Update.
                        temp_flag = False
                        temp_num = temp_num_tr + batch * temp_batch_size
                        if self.update_mode == 'a':
                            for i in range(temp_batch_size):
                                if temp_batch_score[i] > temp_min_score:
                                    temp_flag = True
                                    temp_model.append(temp_batch_score[i])
                                    temp_model_idx.append(temp_num + i)
                        elif self.update_mode == 'r':
                            for i in range(temp_batch_size):
                                if temp_batch_score[i] > temp_min_score:
                                    temp_flag = True
                                    temp_model[temp_min_score_idx] = temp_batch_score[i]
                                    temp_model_idx[temp_min_score_idx] = temp_num + i
                                    temp_min_score_idx = np.argmin(temp_model[temp_num_model[b2b][algorithm]])
                                    temp_min_score = temp_model[temp_min_score_idx]

                        if temp_flag:
                            temp_model_set[b2b][algorithm].append(temp_model)
                            temp_model_idx_set[b2b][algorithm].append(temp_model_idx)
                            temp_num_model[b2b][algorithm] += 1

            # Step 7: Choose the best distance function, algorithm choose type, and classifier.
            best_para = None
            for b2b in self.b2b_type:
                for algorithm in self.algorithm_type:
                    temp_model_idx = temp_model_idx_set[b2b][algorithm]
                    temp_para = self.__get_best_para(temp_model_idx, b2b, temp_tr_idx, temp_va_idx)
                    if best_para is None or temp_para[0] > best_para[0]:
                        best_para = [temp_para[0], temp_para[1], b2b, algorithm, temp_para[-2], temp_para[-1],
                                     temp_model_idx]

            # Best parameters include: f1-measure, number models, distance function, algorithm type, classifier, weights
            yield self.tr_idx[loop], self.te_idx[loop], best_para

    def classify(self):
        """
        Classify.
        """
        temp_sum = 0
        ret_predict = 0
        for tr, te, para in self.self_learning():
            b2b = para[2]
            classifier = para[-3]
            weights = para[-2]
            choose = para[-1]

            temp_predict = []
            temp_tr_lab = self.bags_label[tr]
            temp_te_lab = self.bags_label[te]
            for i in range(len(choose)):
                temp_tr = self.dis[b2b][tr][:, choose[i]]
                temp_te = self.dis[b2b][te][:, choose[i]]
                temp_classify = Classify(get_iter(temp_tr, temp_tr_lab, temp_te, temp_te_lab), para_type=classifier)
                temp_classify.get_result(para_type='f1-measure')
                temp_result = temp_classify.predict
                temp_predict.append((temp_result[0] * weights[i]).tolist())
            temp_predict = np.sum(temp_predict, 0)
            temp_predict[temp_predict >= 0.5] = 1
            temp_predict[temp_predict != 1] = 0

            temp_sum += 1
            temp_tp = np.sum([1 if j == k == 1 else 0 for j, k in zip(temp_predict, temp_te_lab)])
            temp_tn = len(temp_te_lab) + temp_tp - np.sum(
                [1 if j != 1 and k != 1 else 0 for j, k in zip(temp_predict, temp_te_lab)])
            if temp_tn != 0:
                ret_predict += 2 * temp_tp / temp_tn
            else:
                temp_sum -= 1
        return ret_predict / temp_sum
