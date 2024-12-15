import streamlit as st
from retrieve import (
    load_faiss_index,
    load_metadata,
    retrieve_relevant_chunks,
    generate_response,
)
from dotenv import load_dotenv
import os

load_dotenv()

embeddings_folder = os.getenv("EMBEDDINGS_FOLDER")
index_file = os.getenv("INDEX_FILE")
metadata_file = os.getenv("METADATA_FILE")

st.set_page_config(
    page_title="Lorekeeper",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Lorekeeper")

with st.sidebar:
    st.write("# Relevant Chunks")


query = st.text_area("Enter query", "", height=200)

if st.button("Submit"):
    if query.strip():
        index = load_faiss_index()
        metadata = load_metadata()

        relevant_chunks, _ = retrieve_relevant_chunks(query, index, metadata, top_k=5)
        response = generate_response(query, relevant_chunks)

        st.write("## Response")
        st.write(response)

        with st.sidebar:
            if relevant_chunks:
                for chunk in relevant_chunks:
                    st.write(
                        f"**File:** {chunk['file']} | **Page:** {chunk['page_number']} | **Chunk Number:** {chunk['chunk_number']}"
                    )
                    st.write(chunk["chunk"])
                    st.write("---")
            else:
                st.info("No relevant documents found for this query.")
    else:
        st.warning("Please enter a query before generating a response.")
