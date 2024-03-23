import streamlit as st


def system_prompt_page():
    st.title("System Prompt Configuration")
    # Prompt page code goes here
    system_prompt = st.text_area("Enter your prompt")
    if st.button("Save Prompt"):
        st.session_state['system_prompt'] = system_prompt
        st.success("System prompt saved!")


def few_shot_prompt_page():
    # TODO
    pass


# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.vectorstores import LanceDB

# import lancedb

# db = lancedb.connect("/tmp/lancedb")
# table = db.create_table(
#     "my_table",
#     data=[
#         {
#             "vector": embeddings.embed_query("Hello World"),
#             "text": "Hello World",
#             "id": "1",
#         }
#     ],
#     mode="overwrite",
# )

# # Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('../../../state_of_the_union.txt').load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# db = LanceDB.from_documents(documents, OpenAIEmbeddings())

# client = openai.OpenAI(api_key=OPENAI_API_KEY)
# embedding_model_name = "text-embedding-3-small"

# result = client.embeddings.create(
#     input=[
#         "This is a sentence",
#         "A second sentence"
#     ],
#     model=embedding_model_name,
# )

def rag_page():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
    pass