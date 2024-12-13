import streamlit as st
from retrieve import load_faiss_index, load_metadata, retrieve_relevant_chunks, generate_response
from dotenv import load_dotenv
import os

load_dotenv()

embeddings_folder=os.getenv("EMBEDDINGS_FOLDER")
index_file=os.getenv("INDEX_FILE")
metadata_file = os.getenv("METADATA_FILE")

st.set_page_config(
    page_title="Lorekeeper",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Lorekeeper")

col1, col2 = st.columns([1, 2])

with col1:
    query = st.text_area("Enter query", "", height=200)
    
    if st.button("Submit"):
        if query.strip():
            index = load_faiss_index()
            metadata = load_metadata()

            relevant_chunks, _ = retrieve_relevant_chunks(query, index, metadata, top_k=3)
            response = generate_response(query, relevant_chunks)

            st.write("## Response")
            st.write(response)

            with col2:
                st.write("## Relevant Documents")
                if relevant_chunks:
                    for chunk in relevant_chunks:
                        st.write(f"**File:** {chunk['file']} | **Page:** {chunk['page_number']} | **Chunk Number:** {chunk['chunk_number']}")
                        st.write(chunk['chunk'])
                        st.write("---")
                else:
                    st.info("No relevant documents found for this query.")
        else:
            st.warning("Please enter a query before generating a response.")
