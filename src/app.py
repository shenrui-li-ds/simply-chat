import streamlit as st

from chat_page import chat_page
from side_bar import side_bar
from prompt_page import system_prompt_page
from multi_agents_page import multi_agents_page
from rag_page import rag_page

# from audio_recorder_streamlit import audio_recorder
# import whisper
# import numpy as np

def main():
    st.set_page_config(page_title="SimplyChat", page_icon="💬", layout="wide")
    st.session_state.secrets = {}
    if "api_provider" not in st.session_state:
        st.session_state["api_provider"] = None
    
    selected_page = side_bar()
    # Page content based on selection
    if selected_page == "Chat":
        chat_page()
    elif selected_page == "Prompt":
        system_prompt_page()
    elif selected_page == "Multi-Agents":
        multi_agents_page()
    elif selected_page == "RAG":
        rag_page()


if __name__ == "__main__":
    main()
    