import plotly.graph_objects as go
import numpy as np
from keras.models import load_model
from keras.losses import mae
import streamlit as st
from exampleData import ExampleData
import glob
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

name = 'Evaluate'


def app() -> None:
    file_path = model_selector()

    if file_path is None:
        return

    model = model_loader(file_path)

    exampleData = ExampleData()
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = exampleData.get_all_training_data()

    threshold = get_threshold(model, normal_train)
    accuracy, precision, recall = model_evaluation(model, test_data, threshold, exampleData.test_labels)

    plot_metrics(accuracy, precision, recall, threshold)
    plot_loss(model, normal_train, threshold)

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
        return

    selected_data = np.expand_dims(selected_data, axis=0)

    is_normal = predictions(model, selected_data, threshold)

    encoded = model.encoder(selected_data).numpy()
    decoded = model.decoder(encoded).numpy()

    plot_data(selected_label, selected_data, decoded, is_normal)


def plot_metrics(accuracy, precision, recall, threshold):
    acc, prec, rec, thresh = st.columns(4)
    acc.metric("Accuracy", round(accuracy, 2))
    prec.metric("Precision", round(precision, 2))
    rec.metric("Recall", round(recall, 2))
    thresh.metric("Threshold", str(threshold)[:6])


def plot_data(selected_label, selected_data, decoded, is_normal):
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

    st.text(f"Selected: {selected_label}")
    st.text(f"Model predicts: {'Normal' if is_normal else 'Anomaly'}")

    st.plotly_chart(fig, use_container_width=True)


def plot_loss(model, normal_train, threshold):
    reconstructions = model.predict(normal_train)
    train_loss = mae(reconstructions, normal_train)

    fig = go.Figure(data=go.Histogram(x=train_loss, nbinsx=50))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title_text='Train loss',
        yaxis_title_text='Number of examples',
    )

    st.plotly_chart(fig, use_container_width=True)


def model_selector():
    file_paths = {file_path.split('/')[-1]: file_path for file_path in
                  glob.glob('anomalyDetectorApp/model/anomaly-detector-*')}
    option = st.selectbox('Select model', ['<select>'] + list(file_paths.keys()))
    file_path = None

    if option in file_paths:
        file_path = file_paths[option]

    return file_path


def model_loader(file_path):
    """
    Model loader from path
    :param file_path: file path
    :return: loaded model
    """
    return load_model(file_path)


def get_threshold(model, normal_train_data):
    """
    Here we define the threshold to determine whether a data is normal or anomaly.
    Anomaly will be determined when it's above a threshold.
    :param model: trained model
    :param normal_train_data: normal data
    :return: threshold
    """
    reconstructions = model.predict(normal_train_data)
    train_loss = mae(reconstructions, normal_train_data)
    threshold = np.mean(train_loss) + np.std(train_loss)
    return threshold


def predictions(model, data, threshold):
    """
    If the loss is lower than the threshold then normal otherwise anomaly
    :param model: trained model
    :param data: data
    :param threshold: determined threshold
    :return: boolean
    """
    reconstructions = model(data)
    loss = mae(reconstructions, data)
    return tf.math.less(loss, threshold)


def model_evaluation(model, data, threshold, labels):
    """
    Model evaluator
    :param model: trained model
    :param data: test data
    :param threshold: threshold
    :param labels: labels
    :return: accuracy, precision and recall
    """
    preds = predictions(model, data, threshold)
    return accuracy_score(labels, preds), \
           precision_score(labels, preds), \
           recall_score(labels, preds)
