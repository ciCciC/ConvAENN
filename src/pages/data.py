import streamlit as st
import glob
import pandas as pd
import numpy as np
from src.data.dataPreprocessor import DataPreprocessor

name = 'Data'


def app():
    if 'FILE_PATH' not in st.session_state:
        st.session_state['FILE_PATH'] = ''

    if 'DATASET' not in st.session_state:
        st.session_state['DATASET'] = ''

    st.header('Anomaly Detector System with AutoEncoder Model')

    parquet_files = {x.split('/')[-1].split('.')[0]: x for x in glob.glob('src/resources/*_data.parquet')}

    selected_data_set = st.selectbox('Select data set', parquet_files.keys())

    st.session_state['FILE_PATH'] = parquet_files[selected_data_set]
    st.session_state['DATASET_TYPE'] = 'ECG' if selected_data_set.__contains__('ecg') else 'Creditcard'

    st.session_state['DATASET'] = DataPreprocessor(st.session_state['FILE_PATH'])
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = st.session_state['DATASET'].get_all_data()

    shapes = pd.DataFrame({
        'Train': st.session_state['DATASET'].train_data.numpy().shape,
        'Test': st.session_state['DATASET'].test_data.numpy().shape
    }, index=['Rows', 'Columns'])

    unique, counts = np.unique(st.session_state['DATASET'].labels, return_counts=True)
    label_balance = pd.DataFrame({
        'label': unique.astype(int),
        'count': counts
    })

    columns = st.columns(2)
    with columns[0]:
        st.markdown('Data ratio')
        st.dataframe(shapes)
        st.markdown('Label ratio')
        st.dataframe(label_balance)

    with columns[1]:
        w = 200
        h = 200
        idx = 0

        st.text('Normal')
        st.line_chart(normal_train[idx], width=w, height=h)

        st.text('Abnormal')
        st.line_chart(anom_train[idx], width=w, height=h)
