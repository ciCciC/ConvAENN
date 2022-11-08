import pandas as pd
from src.utils.configuration import parquet_engine, credit_data_path, ecg_data_path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class BasePreprocessor(ABC):

    def __init__(self, file_path: str, name: str):
        self.name = name
        self.dataframe = pd.read_parquet(file_path, engine=parquet_engine)
        self.raw_data = self.dataframe.values
        self.labels = None
        self.data = None

    @abstractmethod
    def _normalize(self):
        pass

    def _train_split_data(self):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=21
        )

    def _split_normal_and_anomalies(self):
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


class CreditProcessor(BasePreprocessor):
    """
    Creditcard -> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """

    def __init__(self):
        super().__init__(credit_data_path, 'Credit Card')

        self.labels = 1 - self.raw_data[:, -1]
        self.data = self.raw_data[:, 1:-1]

        self._over_under_sample()
        self._train_split_data()
        self._normalize()
        self._split_normal_and_anomalies()

    def _over_under_sample(self):
        over_sampler = SMOTE(sampling_strategy=.1, n_jobs=-1)
        under_sampler = RandomUnderSampler(sampling_strategy=.3)
        x_over, y_over = over_sampler.fit_resample(self.data, self.labels)
        self.data, self.labels = under_sampler.fit_resample(x_over, y_over)

    def _normalize(self):
        train_amounts = self.train_data[:, -1].reshape(-1, 1)
        test_amounts = self.test_data[:, -1].reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(train_amounts)

        # normalize data
        self.train_data[:, -1] = scaler.transform(train_amounts).squeeze()
        self.test_data[:, -1] = scaler.transform(test_amounts).squeeze()

        # cast to float32
        self.train_data = tf.cast(self.train_data, tf.float32)
        self.test_data = tf.cast(self.test_data, tf.float32)


class ECGProcessor(BasePreprocessor):
    """
    ECG -> https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
    """

    def __init__(self):
        super().__init__(ecg_data_path, 'Electrocardiogram')

        self.labels = self.raw_data[:, -1]
        self.data = self.raw_data[:, :-1]

        self._train_split_data()
        self._normalize()
        self._split_normal_and_anomalies()

    def _normalize(self):
        # obtain min and max values for normalization
        min_val = tf.reduce_min(self.train_data)
        max_val = tf.reduce_max(self.train_data)

        # normalize data
        self.train_data = (self.train_data - min_val) / (max_val - min_val)
        self.test_data = (self.test_data - min_val) / (max_val - min_val)

        # cast to float32
        self.train_data = tf.cast(self.train_data, tf.float32)
        self.test_data = tf.cast(self.test_data, tf.float32)


__data_type_map = {
    'ecg_data': ECGProcessor,
    'creditcard_data': CreditProcessor
}


def data_factory(dataset_type: str) -> BasePreprocessor:
    if dataset_type not in __data_type_map:
        raise FileNotFoundError()

    return __data_type_map[dataset_type]()
