"""
This module is served as main function
"""

import numpy as np
from gbdt import GBClassifier
from training_set import TrainingSet
from sklearn.preprocessing import Imputer

TRAINING_SET_FILE_PATH = 'data/train_set.csv'
TEST_SET_FILE_PATH = 'data/test_set.csv'
OUTPUT_PREDICTION_FILE_PATH = 'data/predicts.csv'


def run():
    training_set = TrainingSet(TRAINING_SET_FILE_PATH)

    training_set_X = training_set.get_X(training_set.float64_training_set)
    training_set_Y = training_set.get_Y(training_set.float64_training_set)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(training_set_X)

    training_set_X = imp.transform(training_set_X)
    training_set_Y = training_set_Y[:, 0]

    gbdt = GBClassifier(training_set_X, training_set_Y, training_set_X)
    gbdt.k_fold_validate(2)
    # print(training_set_Y)


if __name__ == "__main__":
    run()