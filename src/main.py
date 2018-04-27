"""
This module is served as main function
"""

import numpy as np
import pandas as pd
from gbdt import GBClassifier
from dataset import DataSet
from sklearn.preprocessing import Imputer

TRAINING_SET_FILE_PATH = 'data/train_set.csv'
TEST_SET_FILE_PATH = 'data/test_set.csv'
OUTPUT_PREDICTION_FILE_PATH = 'data/predicts.csv'


def remove_multiple_value(test_set):

    def convert(x):
        if x == np.NaN:
            return x
        else:
            return str(x).split(' ')[0]

    test_set['809007'] = test_set['809007'].map(convert)
    test_set['809004'] = test_set['809004'].map(convert)
    test_set['2412'] = test_set['2412'].map(convert)
    test_set['2407'] = test_set['2407'].map(convert)

    test_set['809007'] = pd.to_numeric(test_set['809007'], errors='coerce')
    test_set['809004'] = pd.to_numeric(test_set['809004'], errors='coerce')
    test_set['2412'] = pd.to_numeric(test_set['2412'], errors='coerce')
    test_set['2407'] = pd.to_numeric(test_set['2407'], errors='coerce')

    return test_set.values


def run():
    dataset = DataSet(TRAINING_SET_FILE_PATH, TEST_SET_FILE_PATH)

    training_set_X = dataset.get_X(dataset.float64_training_set)
    training_set_Y = dataset.get_Y(dataset.float64_training_set)

    test_set_X = dataset.get_test_set_X(dataset.test_set, dataset.get_test_columns())

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(training_set_X)

    training_set_X = imp.transform(training_set_X)

    training_set_Y_1 = training_set_Y[:, 0]
    training_set_Y_2 = training_set_Y[:, 1]
    training_set_Y_3 = training_set_Y[:, 2]
    training_set_Y_4 = training_set_Y[:, 3]
    training_set_Y_5 = training_set_Y[:, 4]

    test_set_X = remove_multiple_value(test_set_X)
    test_set_X = imp.transform(test_set_X)

    # print(type(training_set_Y_1[0]))

    gbdt = GBClassifier(training_set_X, training_set_Y_1, test_set_X)
    gbdt.k_fold_validate(2)
    print("\n")

    gbdt = GBClassifier(training_set_X, training_set_Y_2, test_set_X)
    gbdt.k_fold_validate(2)
    print("\n")

    gbdt = GBClassifier(training_set_X, training_set_Y_3, test_set_X, True)
    gbdt.k_fold_validate(2)
    print("\n")

    gbdt = GBClassifier(training_set_X, training_set_Y_4, test_set_X, True)
    gbdt.k_fold_validate(2)
    print("\n")

    gbdt = GBClassifier(training_set_X, training_set_Y_5, test_set_X, True)
    gbdt.k_fold_validate(2)


if __name__ == "__main__":
    run()