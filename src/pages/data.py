import streamlit as st
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import umap
from src.data.basePreprocessor import data_factory

name = 'Data'


def app():
    if 'DATASET' not in st.session_state:
        st.session_state['DATASET'] = ''

    st.header('Anomaly Detection System with AutoEncoder Model')

    parquet_files = [x.split('/')[-1].split('.')[0] for x in glob.glob('src/resources/*_data.parquet')]

    selected_data_set = st.selectbox('Select data set', parquet_files)

    st.session_state['DATASET'] = data_factory(selected_data_set)
    st.session_state['DATASET_TYPE'] = st.session_state['DATASET'].name

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

    expander_stats(shapes, label_balance)
    expander_visualize_single_data_point(normal_train, anom_train)
    expander_visualize_data_points()


def expander_stats(shapes, label_balance):
    with st.expander("Stats"):
        columns = st.columns(2)
        with columns[0]:
            st.markdown('Data ratio')
            st.dataframe(shapes)

        with columns[1]:
            st.markdown('Label ratio')
            st.dataframe(label_balance)


def expander_visualize_single_data_point(normal_train, anom_train):
    with st.expander("Visualize single data"):
        columns = st.columns(2)
        w = 200
        h = 200
        idx = 0

        with columns[0]:
            st.text('Normal')
            st.line_chart(normal_train[idx], width=w, height=h)

        with columns[1]:
            st.text('Anomaly')
            st.line_chart(anom_train[idx], width=w, height=h)


def expander_visualize_data_points():
    with st.expander("Visualize data points"):
        data, labels = st.session_state['DATASET'].get_normalized_data()

        embeddings = umap.UMAP(n_components=2, n_jobs=-1, min_dist=0.0, metric='cosine',
                               random_state=123).fit_transform(data)

        pd_data = pd.DataFrame({
            'e1': embeddings[:, 0],
            'e2': embeddings[:, 1],
            'condition': ['normal 1' if x == 1 else 'anomaly 0' for x in labels.astype(int)]
        })

        plot_data(pd_data)


def plot_data(pd_data: pd.DataFrame):
    fig = px.scatter(
        pd_data,
        x='e1',
        y='e2',
        color='condition'
    )

    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=.2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(
        dict(
            title='Lower dimensional representation',
            plot_bgcolor='black',
            xaxis={'showgrid': False, 'zeroline': False},
            yaxis={'showgrid': False, 'zeroline': False}
        )
    )

    st.plotly_chart(fig, use_container_width=True)
