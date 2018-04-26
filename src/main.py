"""
This module served as main function
Author: Yuzhou
"""

import pandas as pd

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

TRAINING_SET_FILE_PATH = 'data/train_set.csv'
TEST_SET_FILE_PATH = 'data/test_set.csv'
OUTPUT_PREDICTION_FILE_PATH = 'data/predicts.csv'


def build_training_set(training_set_path):
    training_set = pd.read_csv(training_set_path)
    return training_set


def build_test_set():
    return


def text_feature_extract(df):
    """
    Convert the text in dataframe to numerical or boolean value
    :param df: data frame
    :return: data frame
    """
    return df


def run():
    training_set = build_training_set(TRAINING_SET_FILE_PATH)

    header_columns = ['vid', '1', '2', '3', '4', '5']
    column_names = list(training_set)
    test_columns = list(filter(lambda x: x not in header_columns, column_names))

    print(training_set[['004997']])


if __name__ == "__main__":
    run()

