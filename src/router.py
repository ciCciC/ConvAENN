import streamlit as st


class RouterController:

    def __init__(self):
        self.pages = []

        if 'DATASET_TYPE' not in st.session_state:
            st.session_state['DATASET_TYPE'] = ''

    def add_page(self, title, func) -> None:
        self.pages.append({
            'title': title,
            'function': func
        })

    def run(self):
        page = st.sidebar.selectbox(
            'Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )

        page['function']()

        st.sidebar.text(f'Chosen data set: \n {st.session_state["DATASET_TYPE"]}')
