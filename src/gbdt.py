"""
This module is implementation of GBDT classifier.
Author: Yuzhou Yin
"""

import math

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold

LEARNING_RATE = 0.1
MIN_SAMPLES_SPLIT = 300
MIN_SAMPLES_LEAF = 20
MAX_DEPTH = 8
MAX_FEATURES = 'sqrt'
SUBSAMPLE = 0.8
RANDOM_STATE = 10
N_ESTIMATORS = 10


class GBClassifier(object):

    def __init__(self, training_set_X, training_set_Y, test_set):
        self.training_set_X = training_set_X
        self.test_set = test_set
        self.training_set_Y = training_set_Y

    def get_classifier(self, features, labels):
        """
        Return the GBDT classifier fed by the X and Y
        :param features: X
        :param labels: Y
        :return: GBDT classifier model
        """
        gbc = GradientBoostingClassifier(learning_rate=LEARNING_RATE,
                                         min_samples_split=MIN_SAMPLES_SPLIT,
                                         min_samples_leaf=MIN_SAMPLES_LEAF,
                                         max_depth=MAX_DEPTH,
                                         max_features=MAX_FEATURES,
                                         subsample=SUBSAMPLE,
                                         random_state=RANDOM_STATE,
                                         n_estimators=N_ESTIMATORS)
        gbc.fit(features, labels)
        return gbc

    def k_fold_validate(self, fold):
        """
        K-fold validation of the GBDT model
        :param fold: Number of fold
        :return: accuracy, recall, f1_score of each fold
        """
        kf = KFold(n_splits=fold)

        accuracy = []
        recall = []
        f1 = []

        for train_index, test_index in kf.split(self.training_set_X):
            X_train, X_test = self.training_set_X[train_index], self.training_set_X[test_index]
            Y_train, Y_test = self.training_set_Y[train_index], self.training_set_Y[test_index]

            gbc = self.get_classifier(X_train, Y_train)
            y_pred = self.predict(gbc, X_test)

            accuracy.append(accuracy_score(Y_test, y_pred))
            recall.append(recall_score(Y_test, y_pred))
            f1.append(f1_score(Y_test, y_pred))

        print("Average accuracy of %s fold validation: %s" % (fold, sum(accuracy) / len(accuracy)))
        print("Average recall of %s fold validation: %s" % (fold, sum(recall) / len(recall)))
        print("Average f1 score of %s fold validation: %s" % (fold, sum(f1) / len(f1)))

        return accuracy, recall, f1

    @staticmethod
    def meinian_evaluate(y_train, y_pred):
        """
        Evaluation method based on the Meinian description
        :param y_train: training_set labels
        :param y_pred:: predict labels
        :return: evaluation result
        """

        def func(y1, y2):
            r = 0
            for i in range(len(y1)):
                r += math.pow(math.log(y1[i] + 1) - math.log(y2[i] + 1), 2)

            return r / len(y1)

        score = 0
        for i in range(y_train.shape[0]):
            y1 = y_train[i,].tolist()
            y2 = y_pred[i,].tolist()

            score += func(y1, y2)

        print("Score: %s" % score)
        return score

    @staticmethod
    def optimize_parameter(X, y, param_test=None):
        """
        Use grid search to find the best value of given parameters
        :param X: Training set features in use
        :param y: Training set labels in use
        :param param_test: parameters to tune
        :return: the best value of given paramters
        """
        if param_test is None:
            return
        else:
            gsearch = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                        min_samples_leaf=20, max_depth=8,
                                                                        max_features='sqrt',
                                                                        subsample=0.8, random_state=10),
                                   param_grid=param_test, scoring='accuracy_score', iid=False, cv=5)
            gsearch.fit(X, y)
            print("best params: ", gsearch.best_params_)
            print("best score: ", gsearch.best_score_)

            return gsearch.best_params_

    @staticmethod
    def predict(model, X):
        """
        Use the given model and X to predict labels
        :param model: model in use
        :param X: features
        :return: predict labels
        """
        return model.predict(X)
