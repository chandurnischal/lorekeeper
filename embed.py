import os
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from tqdm import tqdm


# Directory paths
data_folder = "data"
embeddings_folder = "embeddings"
index_file = "faiss_index.index"

# Function to read all PDFs and extract text
def read_pdfs(folder):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_reader = PdfReader(os.path.join(folder, file))
            text = "".join([page.extract_text() for page in pdf_reader.pages])
            documents.append((file, text))
    return documents

# Function to chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_text(text)

# Save metadata to JSON
def save_metadata(metadata):
    metadata_path = os.path.join(embeddings_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}.")

# Main embedding script
def main():
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    index_path = os.path.join(embeddings_folder, index_file)
    documents = read_pdfs(data_folder)

    if not documents:
        print("No PDF files found in the data folder.")
        return

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize FAISS index
    embedding_dim = 384
    index = faiss.IndexFlatL2(embedding_dim)

    metadata = []
    all_embeddings = []

    for file, text in tqdm(documents):
        chunks = chunk_text(text)
        embeddings = embeddings_model.embed_documents(chunks)
        index.add(np.array(embeddings, dtype=np.float32))

        metadata.extend({"file": file, "chunk": chunk} for chunk in chunks)
        all_embeddings.extend(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}.")

    # Save metadata
    save_metadata(metadata)

    print("Embeddings and index saved successfully.")

if __name__ == "__main__":
    main()
