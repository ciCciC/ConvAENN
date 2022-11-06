import streamlit as st
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from keras.models import load_model
from keras.losses import mae
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.utils.configuration import metric_file_path, loss_file_path, parquet_engine

name = 'Evaluation'


def app() -> None:
    file_path = model_selector()

    if file_path is None:
        return

    # load model
    model = load_model(file_path)

    # load metric
    df_metric = pd.read_parquet(metric_file_path, engine=parquet_engine)
    model_metric = df_metric[df_metric.Model == file_path].iloc[0, :]

    df_loss = pd.read_parquet(loss_file_path, engine=parquet_engine)
    model_loss = df_loss[df_loss.Model == file_path]

    default_data = st.session_state['DATASET']
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = default_data.get_all_data()

    # Certain datasets might have large test set, in order to not freeze streamlit app we will only use 400
    max_test_data = 400
    normal_test = normal_test[:max_test_data] if normal_test.shape[0] > max_test_data else normal_test
    anom_test = anom_test[:max_test_data] if anom_test.shape[0] > max_test_data else anom_test

    # determine threshold using the normal data
    threshold = model_metric.Threshold
    train_loss = model_loss.Loss

    # evaluate the model given the test data, threshold and test labels
    accuracy, precision, recall, auc = model_metric[['Accuracy', 'Precision', 'Recall', 'AUC']]

    plot_metrics(accuracy, precision, recall, auc, threshold)
    plot_loss(threshold, train_loss)

    normal_selected_data = None
    anomaly_selected_data = None
    selected_data = None
    max_data = 100

    select_col = st.columns(2)
    with select_col[0]:
        normal_selected_data = st.selectbox('Select normal data', [None] + list(range(0, max_data)))
    with select_col[1]:
        anomaly_selected_data = st.selectbox('Select anomaly data', [None] + list(range(0, max_data)))

    if normal_selected_data is not None and anomaly_selected_data is not None:
        # No need to continue on the upcoming code lines
        st.error('Please select normal or anomaly')
        return

    selected_label = None
    if normal_selected_data is not None:
        selected_data = normal_test[normal_selected_data]
        selected_label = 'Normal'
    elif anomaly_selected_data is not None:
        selected_data = anom_test[anomaly_selected_data]
        selected_label = 'Anomaly'
    else:
        # No need to continue on the upcoming code lines
        return

    selected_data = np.expand_dims(selected_data, axis=0)

    is_normal, decoded = predictions(model, selected_data, threshold)

    plot_data(selected_label, selected_data, decoded, is_normal)


def individual_metrics(selected_label, is_normal, decoded, selected_data):
    mean_absolute_error = mae(decoded, selected_data).numpy()[0]
    text_mae = "Mean Absolute Error: {:.4f}".format(mean_absolute_error)
    text_selected = f"Selected: {selected_label}"
    text_predicts = f'Model predicts: {"Normal" if is_normal else "Abnormal"}'

    return f"{text_mae} ||| {text_selected} ||| {text_predicts}"


def plot_metrics(accuracy, precision, recall, auc, threshold):
    acc, prec, rec, roc_auc, thresh = st.columns(5)
    acc.metric("Accuracy", round(accuracy, 2))
    prec.metric("Precision", round(precision, 2))
    rec.metric("Recall", round(recall, 2))
    roc_auc.metric("AUC", round(auc, 2))
    thresh.metric("Threshold", str(threshold)[:6])


def plot_data(selected_label, selected_data, decoded, is_normal):
    title = individual_metrics(selected_label, is_normal, decoded, selected_data)

    x = np.arange(0, selected_data[0].shape[0])
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=selected_data[0],
        name=selected_label
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=decoded[0],
        name='Reconstruction'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=abs(selected_data[0] - decoded[0]),
        name='Error'
    ))

    fig.update_layout(
        dict(
            title=title
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def model_selector():
    file_paths = {file_path.split('/')[-1]: file_path for file_path in
                  glob.glob('./model/anomaly-detector-*')}
    option = st.selectbox('Select model', ['<select>'] + list(file_paths.keys()))
    file_path = None

    if option in file_paths:
        file_path = file_paths[option]

    return file_path


def plot_loss(threshold, train_loss):
    fig = go.Figure(data=go.Histogram(x=train_loss, nbinsx=50))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title_text='Train loss',
        yaxis_title_text='Number of examples',
    )

    st.plotly_chart(fig, use_container_width=True)


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
           roc_auc_score(labels, preds)
