import streamlit as st
from gpt_researcher import GPTResearcher
import asyncio
import os

from contextlib import contextmanager
from io import StringIO
import sys

async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    report = await researcher.run()
    return report

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            buffer.write(b)
            output_func(buffer.getvalue())

        src.write = new_write
        try:
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield
        
def multi_agents_page():
    st.title("Multi-Agents Researcher")
    st.caption("Only support OpenAI and Tavily API at this moment.")

    if "your_api_key" in st.session_state.secrets:
        os.environ["OPENAI_API_KEY"] = st.session_state.secrets["your_api_key"]
    try:
        TAVILY_API_KEY = st.text_input("Your Tavily API Key", value=None, type="password")
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        os.environ["SMART_LLM_MODEL"] = st.session_state["selected_model"]
        os.environ["SMART_TOKEN_LIMIT"] = "4096"
    except TypeError:
        st.warning(' Please Enter your Tavily API key!', icon='⚠️')

    report_types = ["Research Report", "Resource Report", "Outline Report"]
    # Sidebar navigation
    report_type_user = st.radio("Select a report type", report_types)
    report_type_dict = {
        "Research Report": "research_report",
        "Resource Report": "resource_report",
        "Outline Report": "outline_report"
    }

    query = st.text_area("Enter a topic to research (You may use multiple sentences):")
    report_type = report_type_dict[report_type_user]

    if st.button("Start Research"):
        with st.container(height=300):
            with st_stdout("markdown"):
                report = asyncio.run(get_report(query, report_type))

        st.download_button(
            label="Download report as Markdown",
            data=report,
            file_name=f"{report_type}-{'-'.join([w for w in query.split(' ')][:3])}.md",
            mime='text/markdown',
        )
