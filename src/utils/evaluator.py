from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import tensorflow as tf
from keras.losses import mae
import numpy as np
import streamlit as st


def calc_is_normal(reconstructions, data, threshold):
    loss = mae(reconstructions, data)
    return tf.math.less(loss, threshold)


def predictions(model, data, threshold):
    """
    If the loss is lower than the threshold then normal otherwise anomaly
    :param model: trained model
    :param data: data
    :param threshold: determined threshold
    :return: boolean
    """
    reconstructions = model(data)
    return calc_is_normal(reconstructions, data, threshold), reconstructions


def model_evaluation(model, data, threshold, labels):
    """
    Model evaluator
    :param model: trained model
    :param data: test data
    :param threshold: threshold
    :param labels: labels
    :return: accuracy, precision and recall
    """
    preds, reconstructions = predictions(model, data, threshold)
    return accuracy_score(labels, preds), \
           precision_score(labels, preds), \
           recall_score(labels, preds), \
           roc_auc_score(labels, preds), \
           f1_score(labels, preds)


def get_threshold(model, normal_train_data):
    """
    Here we auto define the threshold to determine whether a data is normal or anomaly.
    Anomaly will be determined when it's above a threshold.
    :param model: trained model
    :param normal_train_data: normal data
    :return: threshold
    """
    reconstructions = model.predict(normal_train_data)
    train_loss = mae(reconstructions, normal_train_data)
    # threshold = np.mean(train_loss) + np.std(train_loss)
    threshold = 3 * np.std(train_loss)
    return threshold, train_loss


def plot_metrics(accuracy, precision, recall, f1, auc, threshold):
    acc, prec, rec, f, roc_auc, thresh = st.columns(6)
    acc.metric("Accuracy", round(accuracy, 2))
    prec.metric("Precision", round(precision, 2))
    rec.metric("Recall", round(recall, 2))
    f.metric("F1", round(f1, 2))
    roc_auc.metric("AUC", round(auc, 2))
    thresh.metric("Threshold", str(threshold)[:6])
