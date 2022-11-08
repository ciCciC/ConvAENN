import streamlit as st
import glob
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from src.nnetwork.anomAeModel import AnomalyAeModel
from src.utils.evaluator import model_evaluation, get_threshold, plot_metrics
from src.utils.configuration import metric_file_path, loss_file_path, parquet_engine

name = 'Train'


def app():
    init_states()

    st.header('Train an Anomaly Detection system with AutoEncoder')

    epochs = st.slider('How many epochs to train?', 10, 500, 20)

    data_preprocessor = st.session_state['DATASET']
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = data_preprocessor.get_all_data()

    buttons = st.columns(2)

    with buttons[0]:
        if st.button('Train'):
            train_model(epochs, normal_train, test_data)

            if st.session_state['MODEL'] is not None:
                max_test_data = 1000

                test_data = test_data[:max_test_data] \
                    if test_data.shape[0] > max_test_data \
                    else test_data

                test_labels = data_preprocessor.test_labels[:max_test_data] \
                    if data_preprocessor.test_labels.shape[0] > max_test_data \
                    else data_preprocessor.test_labels

                # Auto determine the threshold
                threshold, train_loss = get_threshold(
                    model=st.session_state['MODEL'], normal_train_data=normal_train)

                # calc metrics
                accuracy, precision, recall, auc, f1 = model_evaluation(
                    model=st.session_state['MODEL'],
                    data=test_data, threshold=threshold, labels=test_labels)

                # populate states
                st.session_state['ACCURACY'] = accuracy
                st.session_state['PRECISION'] = precision
                st.session_state['RECALL'] = recall
                st.session_state['F1'] = f1
                st.session_state['AUC'] = auc
                st.session_state['THRESHOLD'] = threshold
                st.session_state['LOSS'] = train_loss

                st.success('Successfully finished training', icon="✅")

    with buttons[1]:
        save_clicked = st.button('Save?')
        if st.session_state['MODEL'] is not None:
            if save_clicked:
                st.session_state['MODEL_PATH'] = f'./model/anomaly-detector-{st.session_state["DATASET_TYPE"]}-{epochs}'
                st.session_state['MODEL'].save(st.session_state['MODEL_PATH'])

                store_metrics(st.session_state['MODEL_PATH'],
                              st.session_state['ACCURACY'],
                              st.session_state['PRECISION'],
                              st.session_state['RECALL'],
                              st.session_state['F1'],
                              st.session_state['AUC'],
                              st.session_state['THRESHOLD'],
                              st.session_state['LOSS'])

                st.success('Successfully saved', icon="✅")

    plot_metrics(st.session_state['ACCURACY'],
                 st.session_state['PRECISION'],
                 st.session_state['RECALL'],
                 st.session_state['F1'],
                 st.session_state['AUC'],
                 st.session_state['THRESHOLD'])

    plot_loss(epochs)


def init_states():
    if 'HISTORY' not in st.session_state:
        st.session_state['HISTORY'] = {'train_loss': [0], 'val_loss': [0]}
    if 'MODEL' not in st.session_state:
        st.session_state['MODEL'] = None
    if 'MODEL_PATH' not in st.session_state:
        st.session_state['MODEL_PATH'] = None
    if 'ACCURACY' not in st.session_state:
        st.session_state['ACCURACY'] = 0
    if 'PRECISION' not in st.session_state:
        st.session_state['PRECISION'] = 0
    if 'RECALL' not in st.session_state:
        st.session_state['RECALL'] = 0
    if 'F1' not in st.session_state:
        st.session_state['F1'] = 0
    if 'AUC' not in st.session_state:
        st.session_state['AUC'] = 0
    if 'THRESHOLD' not in st.session_state:
        st.session_state['THRESHOLD'] = 0
    if 'LOSS' not in st.session_state:
        st.session_state['LOSS'] = None


def train_model(epochs, normal_train, test_data):
    optimizer = Adam()
    loss = 'mae'
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    units = normal_train.shape[-1]

    model = AnomalyAeModel(units=units)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(normal_train, normal_train,
                        epochs=epochs,
                        batch_size=512,
                        validation_data=(test_data, test_data),
                        shuffle=True,
                        callbacks=[early_stop])

    st.session_state['HISTORY'] = {'train_loss': history.history["loss"], 'val_loss': history.history["val_loss"]}
    st.session_state['MODEL'] = model


def store_metrics(model_path, accuracy, precision, recall, f1, auc, threshold, train_loss):
    # check existence
    metric_file = glob.glob(metric_file_path)
    loss_file = glob.glob(loss_file_path)

    df_metric = None
    df_loss = None

    new_metric_data = {
        'Model': [model_path],
        'Accuracy': [round(accuracy, 2)],
        'Precision': [round(precision, 2)],
        'Recall': [round(recall, 2)],
        'F1': [round(f1, 2)],
        'AUC': [round(auc, 2)],
        'Threshold': [float(str(threshold)[:6])]
    }

    new_loss_data = {
        'Model': [model_path] * len(train_loss),
        'Loss': train_loss
    }

    if len(metric_file) == 0 and len(loss_file) == 0:
        df_metric = pd.DataFrame(new_metric_data)
        df_loss = pd.DataFrame(new_loss_data)

        df_metric.to_parquet(metric_file_path, index=False, engine=parquet_engine)
        df_loss.to_parquet(loss_file_path, index=False, engine=parquet_engine)
    else:
        df_metric = pd.read_parquet(metric_file[0], engine=parquet_engine)
        df_loss = pd.read_parquet(loss_file[0], engine=parquet_engine)

        does_metric_model_exist = df_metric[df_metric.Model == model_path].shape[0] > 0
        does_loss_model_exist = df_loss[df_loss.Model == model_path].shape[0] > 0

        # populate dataframes when model does not exist
        if not (does_metric_model_exist and does_loss_model_exist):
            df_metric = pd.concat([df_metric, pd.DataFrame(new_metric_data)], ignore_index=True, axis=0)
            df_loss = pd.concat([df_loss, pd.DataFrame(new_loss_data)], ignore_index=True, axis=0)

            df_metric.to_parquet(metric_file_path, index=False, engine=parquet_engine)
            df_loss.to_parquet(loss_file_path, index=False, engine=parquet_engine)


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
