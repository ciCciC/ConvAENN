import streamlit as st
from random import random
from src.nnetwork.anomAeModel import AnomalyAeModel
from keras.optimizers import Adam
from src.data.exampleData import ExampleData
import plotly.graph_objects as go
import numpy as np
from src.pages.evaluation import model_evaluation, get_threshold, plot_metrics

name = 'Train'


def app():
    if 'HISTORY' not in st.session_state:
        st.session_state['HISTORY'] = {'train_loss': [0], 'val_loss': [0]}

    if 'MODEL' not in st.session_state:
        st.session_state['MODEL'] = None

    st.header('Train an Anomaly Detector with AutoEncoder')

    epochs = st.slider('How many epochs to train?', 10, 100, 20)

    exampleData = ExampleData()
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = exampleData.get_all_training_data()

    with st.container():
        buttons = st.columns(2)

        with buttons[0]:
            st.button('Train', key=random(), on_click=train_model, args=(epochs, train_data, normal_train, test_data))

        with buttons[1]:
            save_clicked = st.button('Save?')
            if st.session_state['MODEL'] is not None:
                if save_clicked:
                    st.session_state['MODEL'].save(f'./model/anomaly-detector-{epochs}')
                    st.success('Successfully saved', icon="✅")

    if st.session_state['MODEL'] is not None:
        plot_metric(st.session_state['MODEL'], normal_train, test_data, exampleData.test_labels)
        plot_loss(epochs)


def train_model(epochs, train_data, normal_train, test_data):
    optimizer = Adam()
    loss = 'mae'

    units = train_data.shape[-1]

    model = AnomalyAeModel(units=units)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(normal_train, normal_train,
                        epochs=epochs,
                        batch_size=512,
                        validation_data=(test_data, test_data),
                        shuffle=True)

    st.session_state['HISTORY'] = {'train_loss': history.history["loss"], 'val_loss': history.history["val_loss"]}
    st.session_state['MODEL'] = model


def plot_metric(model, normal_train, test_data, test_labels):
    threshold = get_threshold(model, normal_train)
    accuracy, precision, recall, auc = model_evaluation(model, test_data, threshold, test_labels)
    plot_metrics(accuracy, precision, recall, auc, threshold)


def plot_loss(epochs):
    x = np.arange(0, epochs)
    fig = go.Figure(data=go.Line(
        x=x,
        y=st.session_state['HISTORY']['train_loss'],
        name='Train loss'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=st.session_state['HISTORY']['val_loss'],
        name='Validation loss'
    ))

    layout = dict(
        title='Loss over epochs',
        xaxis=dict(title="Epoch", showgrid=False),
        yaxis=dict(title="Loss", showgrid=False)
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)
