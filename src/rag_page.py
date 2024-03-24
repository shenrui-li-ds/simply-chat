from langchain_openai import OpenAIEmbeddings
from lancedb import Lance
import streamlit as st
import openai

# Initialize OpenAI Embeddings
openai_api_key = st.session_state.secrets["your_api_key"]
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize Lance for your database
lance_db = Lance("your_database_configuration")

def rag_retrieve_and_generate(query):
    # Use embeddings to find relevant information in LanceDB
    relevant_docs = lance_db.retrieve(query, embeddings)

    # Use a suitable LLM for generation based on the retrieved docs
    combined_context = " ".join([doc.content for doc in relevant_docs]) + query
    response = openai.Completion.create(engine="davinci", prompt=combined_context, max_tokens=150)

    return response.choices[0].text


def rag_page():
    st.title("Retrieve and Generate (RAG)")
    user_query = st.text_input("Enter your query:")

    if st.button("Generate"):
        with st.spinner("Retrieving and generating..."):
            generated_response = rag_retrieve_and_generate(user_query)
        st.text_area("Generated Response", value=generated_response, height=300)


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