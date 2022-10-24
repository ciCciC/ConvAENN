import streamlit as st

import evaluation
import home
import train
from router import RouterController

st.set_page_config(page_title='Anomaly Detector System', layout="wide")

app = RouterController()
app.add_page(home.name, home.app)
app.add_page(train.name, train.app)
app.add_page(evaluation.name, evaluation.app)
app.run()
