"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建日期: 2021 0303
近一次修改: 2021 0715
"""

import warnings
warnings.filterwarnings("ignore")


class Classify:
    """
    调用sklearn库中的分类器，实现多示例学习的映射向量分类
    :param
        classifier_type：    所使用分类器的列表，如["knn"]
                             已有的分类器包括"knn"、"j48"、"svm"，所调用的分类器的参数均使用默认参数
        performance_type：   性能度量指标列表，如["f1_score"]
                             已有的性能度量包括"f1_score"、"acc"、"roc"
        当参数均使用None时，将使用默认分类器和默认分类性能度量
    :attribute
        tr_true_label_arr：  训练集真实标签
        tr_predict_arr：     训练集测试标签
        te_true_label_arr：  测试集真实标签
        te_predict_arr：     测试集预测标签
        tr_per：             训练集分类性能
        te_per：             测试集分类性能
    """

    def __init__(self, classifier_type=None, performance_type=None):

        self.__classifier_type = classifier_type
        self.__performance_type = performance_type
        self.tr_true_label_arr = {}
        self.tr_predict_arr = {}
        self.te_true_label_arr = {}
        self.te_predict_arr = {}
        self.tr_per = {}
        self.te_per = {}
        self.__init_classify()

    def __init_classify(self):
        """
        分类器初始化
        """

        self.__classifier = []
        self.__performance_er = []
        if self.__classifier_type is None:
            self.__classifier_type = ["knn"]
        for classifier_type in self.__classifier_type:
            if classifier_type == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                self.__classifier.append(KNeighborsClassifier(n_neighbors=3))
            elif classifier_type == "svm":
                from sklearn.svm import SVC
                self.__classifier.append(SVC(max_iter=10000))
            elif classifier_type == "j48":
                from sklearn.tree import DecisionTreeClassifier
                self.__classifier.append(DecisionTreeClassifier())

        if self.__performance_type is None:
            self.__performance_type = ["f1_score"]
        for performance_type in self.__performance_type:
            if performance_type == "f1_score":
                from sklearn.metrics import f1_score
                self.__performance_er.append(f1_score)
            elif performance_type == "acc":
                from sklearn.metrics import accuracy_score
                self.__performance_er.append(accuracy_score)
            elif performance_type == "roc":
                from sklearn.metrics import roc_auc_score
                self.__performance_er.append(roc_auc_score)

    def __reset_record(self):
        """
        重设记录向量
        """
        for classifier_type in self.__classifier_type:
            self.tr_predict_arr[classifier_type], self.tr_true_label_arr[classifier_type] = [], []
            self.tr_per[classifier_type] = []
            self.te_predict_arr[classifier_type], self.te_true_label_arr[classifier_type] = [], []
            self.te_per[classifier_type] = []

    def test(self, data_iter, is_pre_tr=False):
        """
        :param
            data_iter：          数据迭代器
            is_pre_tr：          是否需要预测训练集
        """
        self.__reset_record()
        for tr_data, tr_label, te_data, te_label in data_iter:
            for classifier, classifier_type in zip(self.__classifier, self.__classifier_type):
                model = classifier.fit(tr_data, tr_label)

                if is_pre_tr:
                    predict = model.predict(tr_data)
                    self.tr_predict_arr[classifier_type].extend(predict)
                    self.tr_true_label_arr[classifier_type].extend(tr_label)

                predict = model.predict(te_data)
                self.te_predict_arr[classifier_type].extend(predict)
                self.te_true_label_arr[classifier_type].extend(te_label)

        for classifier_type in self.__classifier_type:
            for per_er in self.__performance_er:
                try:
                    self.tr_per[classifier_type].append(per_er(
                        self.tr_predict_arr[classifier_type],
                        self.tr_true_label_arr[classifier_type]
                    ))
                    self.te_per[classifier_type].append(per_er(
                        self.te_predict_arr[classifier_type],
                        self.te_true_label_arr[classifier_type]
                    ))
                except ValueError:
                    self.tr_per[classifier_type].append(0)
                    self.te_per[classifier_type].append(0)

        if is_pre_tr:
            return self.tr_per, self.te_per
        return self.te_per
