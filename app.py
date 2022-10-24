import streamlit as st

from src.pages import train, evaluation, home
from src.router import RouterController


def config():
    st.set_page_config(page_title='Anomaly Detector System', layout="wide")


def main():
    config()
    app = RouterController()
    app.add_page(home.name, home.app)
    app.add_page(train.name, train.app)
    app.add_page(evaluation.name, evaluation.app)
    app.run()


if __name__ == '__main__':
    main()
