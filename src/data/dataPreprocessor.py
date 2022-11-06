import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.configuration import parquet_engine


class DataPreprocessor:
    """
    ECG -> https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
    Creditcard -> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """

    def __init__(self, file_path: str):
        is_ecg = file_path.__contains__('ecg')

        self.dataframe = pd.read_parquet(file_path, engine=parquet_engine)
        self.raw_data = self.dataframe.values
        self.labels = self.raw_data[:, -1] if is_ecg else 1 - self.raw_data[:, -1]
        self.data = self.raw_data[:, :-1] if is_ecg else self.raw_data[:, 1:-1]

        self.__train_split_data()
        self.__normalize()
        self.__split_normal_and_anomalies()

    def __train_split_data(self):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=21
        )

    def __normalize(self):
        # obtain min and max values for normalization
        min_val = tf.reduce_min(self.train_data)
        max_val = tf.reduce_max(self.train_data)

        # normalize data
        self.train_data = (self.train_data - min_val) / (max_val - min_val)
        self.test_data = (self.test_data - min_val) / (max_val - min_val)

        # cast to float32
        self.train_data = tf.cast(self.train_data, tf.float32)
        self.test_data = tf.cast(self.test_data, tf.float32)

    def __split_normal_and_anomalies(self):
        # transform to boolean
        train_labels = self.train_labels.astype(bool)
        test_labels = self.test_labels.astype(bool)

        # subset normal data
        self.normal_train = self.train_data[train_labels]
        self.normal_test = self.test_data[test_labels]

        # subset anomaly data
        self.anom_train = self.train_data[~train_labels]
        self.anom_test = self.test_data[~test_labels]

    def get_all_data(self):
        return self.train_data, self.test_data, self.normal_train, self.normal_test, self.anom_train, self.anom_test
