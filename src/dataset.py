"""
This module is class of training set
Author: Yuzhou
"""

import pandas as pd
import numpy as np

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

TRAINING_SET_FILE_PATH = 'data/train_set.csv'
TEST_SET_FILE_PATH = 'data/test_set.csv'
OUTPUT_PREDICTION_FILE_PATH = 'data/predicts.csv'


class DataSet(object):

    def __init__(self, training_set_path, test_set_path):
        self.training_set_path = training_set_path
        self.test_set_path = test_set_path

        self.training_set = self.build_training_set()
        self.test_set = self.build_test_set()

        self.float64_training_set = self.select_numeric_data(self.training_set)

        self.header_columns = ['1', '2', '3', '4', '5']

    def build_test_set(self):
        return pd.read_csv(self.test_set_path)

    def build_training_set(self):
        return pd.read_csv(self.training_set_path, dtype={'1': np.float64,
                                                          '2': np.float64,
                                                          '3': np.float64,
                                                          '4': np.float64,
                                                          '5': np.float64})

    def text_feature_extract(df):
        """
        Convert the text in dataframe to numerical or boolean value
        :param df: data frame
        :return: data frame
        """
        return df

    @staticmethod
    def select_numeric_data(df):
        return df.select_dtypes(['float64'])

    def get_Y(self, dataset):
        return dataset[self.header_columns].values

    def get_X(self, dataset):
        return dataset.drop(self.header_columns, axis=1).values

    def get_test_set_X(self, test_set, columns):
        return test_set[columns]

    def get_test_columns(self):
        return list(self.float64_training_set.drop(self.header_columns, axis=1))


if __name__ == "__main__":
    dataset = DataSet(TRAINING_SET_FILE_PATH, TEST_SET_FILE_PATH)