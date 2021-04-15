"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0713, last modified in 2020 1116.
@note: The classify algorithms, the given data must be a iterator, and 
    its including training set, training label, test set and test label.
"""


import numpy as np


class Classify:
    """
    The classify algorithms come from sklearn.
    :param
        para_data:
            The given data, and its must be including training set, training label, test set and test label.
        para_type:
            The type of  classify, and the default setting is 'knn', and others include
                               'random_forest',
                               'decision_tree',
                               'SVM',
                               'logistic_regression'
                               'ada_boost'
                               'gaussian_NB'
                               'linear_discriminant'
                               'multinomial_NB'
    @attribute
        data:
            The training / test with label.
        type:
            The given type of classification algorithm.
        type_list:
            The type of algorithms.
        result_type_list:
            The type of classification evuation
    """

    def __init__(self, para_data, para_type='knn'):
        """
        The constructor.
        """
        self.data = para_data
        self.type = para_type
        self.predict = {}
        self.type_list = ['knn',
                          'random_forest',
                          'j48',
                          'svm',
                          'logistic_regression'
                          'ada_boost'
                          'gaussian_NB'
                          'linear_discriminant'
                          'multinomial_NB']
        self.result_type_list = ['accuracy',
                                 'precision',
                                 'recall',
                                 'f1-measure']

    def get_result(self, para_k=3, para_type='accuracy'):
        """
        Get accuracy using some classify algorithm.
        :param
            para_k:
                The number of neighbors for knn, and its default setting is 3.
            para_type:
                The type of the classification evaluation metric, ant its default setting is accuracy;
                The other setting includes:
                    precision,
                    recall,
                    f1-measure.
        :return:
            The accuracy.
        @note:
            In addition to accuracy, other indicators can only be applied to binary
        """
        if self.type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            temp_classify = KNeighborsClassifier(para_k, p=1)
        elif self.type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            temp_classify = RandomForestClassifier(n_estimators=10)
        elif self.type == 'j48':
            from sklearn import tree
            temp_classify = tree.DecisionTreeClassifier()
        elif self.type == 'svm':
            from sklearn.svm import SVC
            temp_classify = SVC(kernel='poly')
        elif self.type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            temp_classify = LogisticRegression(penalty='l2')
        elif self.type == 'ada_boost':
            from sklearn.ensemble import AdaBoostClassifier
            temp_classify = AdaBoostClassifier()
        elif self.type == 'gaussian_NB':
            from sklearn.naive_bayes import GaussianNB
            temp_classify = GaussianNB()
        elif self.type == 'linear_discriminant':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            temp_classify = LinearDiscriminantAnalysis()
        elif self.type == 'multinomial_NB':
            from sklearn.naive_bayes import MultinomialNB
            temp_classify = MultinomialNB()
        else:
            raise ("Fatal error, there have not %s algorithm." % self.type)

        temp_num = 0
        temp_sum = 0
        loop = 0

        for training_data, training_label, test_data, test_label in self.data:
            temp_classify.fit(training_data, training_label)
            temp_sum += 1
            self.predict[loop] = temp_classify.predict(test_data)
            if para_type == 'accuracy':
                temp_predict = [1 if value else 0 for value in self.predict[loop] == test_label]
                temp_num += np.sum(temp_predict) / len(temp_predict)
            else:
                temp_tp = np.sum([1 if i == j == 1 else 0 for i, j in zip(self.predict[loop], test_label)])
                if para_type == 'precision':
                    temp_pre = np.sum(self.predict[loop][self.predict[loop] == 1])
                    if temp_pre != 0:
                        temp_num += temp_tp / temp_pre
                    else:
                        temp_sum -= 1
                elif para_type == 'recall':
                    temp_pre = np.sum(test_label[test_label == 1])
                    if temp_pre != 0:
                        temp_num += temp_tp / temp_pre
                    else:
                        temp_sum -= 1
                elif para_type == 'f1-measure':
                    temp_tn = len(test_label) + temp_tp - \
                              np.sum([1 if i != 1 and j != 1 else 0 for i, j in zip(self.predict[loop], test_label)])
                    if temp_tn != 0:
                        temp_num += 2 * temp_tp / temp_tn
                    else:
                        temp_sum -= 1
                loop += 1

        return temp_num / temp_sum if temp_sum != 0 else 0
