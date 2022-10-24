import streamlit as st
import pandas as pd

from src.data.exampleData import ExampleData

name = 'Data'


def app():
    st.header('Anomaly Detector System with AutoEncoder Model')
    st.subheader('ECG data published by Google')

    exampleData = ExampleData()
    train_data, test_data, normal_train, normal_test, anom_train, anom_test = exampleData.get_all_training_data()

    shapes = pd.DataFrame({
        'Train': exampleData.train_data.numpy().shape,
        'Test': exampleData.test_data.numpy().shape
    }, index=['Rows', 'Columns'])

    columns = st.columns(2)
    with columns[0]:
        st.dataframe(shapes)

    w = 200
    h = 200

    with columns[1]:
        st.text('Normal ECG')
        st.line_chart(normal_test[0], width=w, height=h)

        st.text('Anomaly ECG')
        st.line_chart(anom_test[0], width=w, height=h)


