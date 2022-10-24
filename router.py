import streamlit as st


class RouterController:

    def __init__(self):
        self.pages = []

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
