import streamlit as st
import os
import tempfile

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader, 
    PythonLoader,
    TextLoader, 
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    # UnstructuredImageLoader,
    UnstructuredMarkdownLoader, 
    UnstructuredWordDocumentLoader, 
)

# Custom loader class to set utf-8 encoding
class CustomTextLoader(TextLoader):
    def __init__(self, file_path, encoding='utf-8'):
        super().__init__(file_path)
        self.encoding = encoding


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
        # ".jpg": UnstructuredImageLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        # ".png": UnstructuredImageLoader,
        ".py": PythonLoader,
        ".txt": CustomTextLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    # Retrieve the loader class based on the file extension
    loader_class = loaders.get(file_extension)

    if not loader_class:
        raise ValueError(f"Unsupported file type: \"{file_extension}\". Please remove this file to proceed.")
    
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
            for idx, text in enumerate(file_knowledge):
                text.metadata["id"] = idx
            knowledge.extend(file_knowledge)
            os.unlink(temp_file_path)  # Remove the temporary file
        except ValueError as e:
            st.warning(str(e))
            continue

    return knowledge

def base_rag_chain(knowledge, user_query, api_key, 
                   embedding_model_class, embedding_model_name, 
                   chat_model_class, chat_model_name):
    # Define the embedding model
    api_provider = st.session_state["api_provider"]
    key_param = {
        "Mistral": "mistral_api_key",
        "OpenAI": "openai_api_key",
    }
    # Ensure the provider is supported, else raise an exception
    if api_provider in key_param:
        # Providers that require an API key
        api_param = {key_param[api_provider]: api_key}
    elif api_provider == "Ollama":
        # "Ollama" does not require an API key
        api_param = {}
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")
    
    embeddings = embedding_model_class(
        model=embedding_model_name, 
        **api_param,
    )

    # if api_provider == "Mistral":
    #     embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    # elif api_provider == "OpenAI":
    #     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    # Create the vector store 
    vector = FAISS.from_documents(knowledge, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()

    # Define the chat model
    temperature = st.session_state["temperature"]
    max_tokens = st.session_state["max_tokens"]
    model_params = {
        'temperature': temperature,
        # 'max_tokens': max_tokens
    }
    chat_model = chat_model_class(
        model=chat_model_name, 
        **api_param,
        **model_params,
    )

    # if api_provider == "Mistral":
    #     chat_model = ChatMistralAI(model=chat_model_name, mistral_api_key=api_key, **model_params)
    # elif api_provider == "OpenAI":
    #     chat_model = ChatOpenAI(model=chat_model_name, openai_api_key=api_key, **model_params)

    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
    """Based on the provided context, please answer the following question. If the answer isn't found within the context, kindly state 'information not found' and suggest a plausible alternative if possible.

    <context>
    {context}
    </context>

    Query: {input}"""
    )

    # # Basic retrieval chain to answer questions
    # document_chain = create_stuff_documents_chain(model, prompt)
    # basic_retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Reranking for better retrieval precision
    # Flashrank reranker
    compressor = FlashrankRerank(client=Ranker, top_n=4, model="rank-T5-flan")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )

    document_chain = create_stuff_documents_chain(chat_model, prompt)
    rerank_retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    # Custom defined function to stream response to UI
    def generate_responses():
        # for chunk in retrieval_chain.stream({"query": user_query}):
        for chunk in rerank_retrieval_chain.stream({"input": user_query}):
            # print(chunk)
            if 'answer' in chunk:
                yield chunk['answer']
            elif 'result' in chunk:
                yield chunk['result']
    # Stream response to streamlit
    response = st.write_stream(generate_responses)
    st.session_state.messages.append({"role": "assistant", "content": response})

def mistral_rag(knowledge, user_query, mistral_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, mistral_api_key, 
                   MistralAIEmbeddings, "mistral-embed", 
                   ChatMistralAI, chat_model_name)

def openai_rag(knowledge, user_query, openai_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, openai_api_key, 
                   OpenAIEmbeddings, "text-embedding-3-small", 
                   ChatOpenAI, chat_model_name)
    
def ollama_rag(knowledge, user_query, ollama_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, ollama_api_key, 
                   OllamaEmbeddings, "mxbai-embed-large", 
                   ChatOllama, chat_model_name)

def rag_page():
    st.title("Retrieval-Augmented Generation (RAG)")
    st.caption("Only support OpenAI, Mistral and Ollama models at this time.")
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

                    elif rag_api_provider == "Ollama":
                        ollama_api_key = ""
                        with st.spinner("Retrieving and generating..."):
                            ollama_rag(knowledge, user_query, ollama_api_key)
                    
                    else:
                        st.warning(' Selected provider is not supported at this time.', icon='⚠️')

        else:
            st.warning(' Failed to parse uploaded document(s).', icon='⚠️')

