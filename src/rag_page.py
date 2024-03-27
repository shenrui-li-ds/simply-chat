import streamlit as st
import os
import tempfile

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader, 
    PythonLoader,
    TextLoader, 
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader, 
    UnstructuredWordDocumentLoader, 
)

# import inspect
# from langchain_community import document_loaders

# def get_loader(file_path):
#     _, file_extension = os.path.splitext(file_path)
#     file_extension = file_extension.lower().lstrip('.')  # Remove dot and convert to lowercase

#     # Convert file extension to loader class name following a convention
#     loader_class_name = f"{file_extension.capitalize()}Loader"

#     # Search for a class with the matching name in the document_loaders module
#     for name, obj in inspect.getmembers(document_loaders, inspect.isclass):
#         if name == loader_class_name:
#             return obj(file_path)  # Instantiate and return the loader class

#     raise ValueError(f"No loader found for file type: {file_extension}")

def get_loader(file_path):
    # Map of file extensions to their corresponding loader classes
    loaders = {
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".html": UnstructuredHTMLLoader,
        ".json": JSONLoader,
        ".jpg": UnstructuredImageLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".png": UnstructuredImageLoader,
        ".py": PythonLoader,
        ".txt": TextLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Retrieve the loader class based on the file extension
    loader_class = loaders.get(file_extension)

    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader_class

@st.cache_data
def file_processor(uploaded_files):
    knowledge = []
    text_splitter = RecursiveCharacterTextSplitter()
    
    for uploaded_file in uploaded_files:
        try:
            loader_class = get_loader(uploaded_file.name)
            # Create tempfile because loader_class takes PathLike objects
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = loader_class(file_path=temp_file_path)
            docs = loader.load()
            # Split text into chunks
            file_knowledge = text_splitter.split_documents(docs)
            knowledge.extend(file_knowledge)
            os.unlink(temp_file_path)  # Remove the temporary file
        except ValueError as e:
            st.warning(str(e))
            continue

    return knowledge

def mistral_rag(knowledge, user_query, mistral_api_key):
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=mistral_api_key)
    # Create the vector store 
    vector = FAISS.from_documents(knowledge, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    model = ChatMistralAI(mistral_api_key=mistral_api_key)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
    """Based on the provided context, please answer the following question. If the answer isn't found within the context, kindly state 'information not found' and suggest a plausible alternative if possible.

    <context>
    {context}
    </context>

    Query: {input}"""
    )

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # response = retrieval_chain.invoke({"input": user_query})
    def generate_responses():
        for chunk in retrieval_chain.stream({"input": user_query}):
            # print(chunk)
            if 'answer' in chunk:
                yield chunk['answer']
    # Stream response to streamlit
    response = st.write_stream(generate_responses)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # return response["answer"]

def openai_rag(knowledge, user_query, openai_api_key):
    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    # Create the vector store 
    vector = FAISS.from_documents(knowledge, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    model = ChatOpenAI(openai_api_key=openai_api_key)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
    """Based on the provided context, please answer the following question. If the answer isn't found within the context, kindly state 'information not found' and suggest a plausible alternative if possible.

    <context>
    {context}
    </context>

    Query: {input}"""
    )

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    def generate_responses():
        for chunk in retrieval_chain.stream({"input": user_query}):
            # print(chunk)
            if 'answer' in chunk:
                yield chunk['answer']
    # Stream response to streamlit
    response = st.write_stream(generate_responses)
    st.session_state.messages.append({"role": "assistant", "content": response})

def rag_page():
    st.title("Retrieval-Augmented Generation (RAG)")
    st.caption("Only support OpenAI and Mistral models at this time.")
    if "api_provider" in st.session_state and "selected_model" in st.session_state:
        api_provider = st.session_state["api_provider"]
        selected_model = st.session_state["selected_model"]
        st.caption((
            f"Current model: [{api_provider}] - [{selected_model}]. "
            "LLM can make mistakes. Consider checking important information."
        ))
    else:
        st.caption((
            "No API provider or model selected. "
            "LLM can make mistakes. Consider checking important information."
        ))
    uploaded_files = st.file_uploader("Upload your file(s)", accept_multiple_files=True)
    rag_api_provider = st.session_state["api_provider"]

    if "your_api_key" not in st.session_state.secrets:
        st.warning(' Please enter your credentials in the side bar first.', icon='⚠️')

    if uploaded_files:
        knowledge = file_processor(uploaded_files)
    
        if knowledge:
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
                # st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Enter your query here"):
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.messages.append({"role": "user", "content": user_query})

                with st.chat_message("assistant"):
                    if rag_api_provider == "Mistral":
                        mistral_api_key = st.session_state.secrets["your_api_key"]
                        with st.spinner("Retrieving and generating..."):
                            mistral_rag(knowledge, user_query, mistral_api_key)

                    elif rag_api_provider == "OpenAI":
                        openai_api_key = st.session_state.secrets["your_api_key"]
                        with st.spinner("Retrieving and generating..."):
                            openai_rag(knowledge, user_query, openai_api_key)
                    
                    else:
                        st.warning(' Only support OpenAI and Mistral models at this time.', icon='⚠️')

        else:
            st.warning(' Failed to parse uploaded document(s).', icon='⚠️')

