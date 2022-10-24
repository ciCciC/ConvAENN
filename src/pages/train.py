import streamlit as st
from random import random
from src.nnetwork.anomAeModel import AnomalyAeModel
from keras.optimizers import Adam
from src.data.exampleData import ExampleData
import plotly.graph_objects as go
import numpy as np

name = 'Train'


def app():
    if 'HISTORY' not in st.session_state:
        st.session_state['HISTORY'] = {'train_loss': [0], 'val_loss': [0]}

    st.header('Train an Anomaly Detector with AutoEncoder')

    epochs = st.slider('How many epochs to train?', 10, 100, 20)

    with st.container():
        buttons = st.columns(2)

        with buttons[0]:
            to_save = st.checkbox('Save network')

        with buttons[1]:
            st.button('Train', key=random(), on_click=train_model, args=(epochs, to_save))

    plot(epochs)


def train_model(epochs, to_save):
    optimizer = Adam()
    loss = 'mae'

    exampleData = ExampleData()

    train_data, test_data, normal_train, normal_test, anom_train, anom_test = exampleData.get_all_training_data()

    units = train_data.shape[-1]

    model = AnomalyAeModel(units=units)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(normal_train, normal_train,
                        epochs=epochs,
                        batch_size=512,
                        validation_data=(test_data, test_data),
                        shuffle=True)

    if to_save:
        model.save(f'anomalyDetectorApp/model/anomaly-detector-{epochs}')
        st.success('Successfully saved', icon="✅")

    st.session_state['HISTORY'] = {'train_loss': history.history["loss"], 'val_loss': history.history["val_loss"]}


def plot(epochs):
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
