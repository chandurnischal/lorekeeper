import os
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import fitz
from tqdm import tqdm

data_folder = "data"
embeddings_folder = "embeddings"
index_file = "faiss_index.index"

def read_pdfs(folder):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            try:
                with fitz.open(file_path) as pdf:
                    document_pages = [
                        (page.number + 1, page.get_text())
                        for page in pdf
                    ]
                    documents.append((file, document_pages))
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return documents

def chunk_text_with_metadata(document_pages, file_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks_metadata = []
    for page_number, page_text in tqdm(document_pages, desc="Pages processed..."):
        chunks = text_splitter.split_text(page_text)
        for chunk_number, chunk in enumerate(chunks, start=1):
            chunks_metadata.append({
                "file": file_name,
                "page_number": page_number,
                "chunk_number": chunk_number,
                "chunk": chunk
            })
    return chunks_metadata

def save_metadata(metadata):
    metadata_path = os.path.join(embeddings_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}.")

if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

index_path = os.path.join(embeddings_folder, index_file)

documents = read_pdfs(data_folder)

if not documents:
    print("No PDF files found in the data folder.")
    exit()

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

metadata = []
all_embeddings = []

for file, document_pages in documents:
    chunks_metadata = chunk_text_with_metadata(document_pages, file)
    for chunk_metadata in chunks_metadata:
        chunk_text = chunk_metadata["chunk"]
        embedding = embeddings_model.embed_query(chunk_text)
        index.add(np.array([embedding], dtype=np.float32))

        metadata.append(chunk_metadata)
        all_embeddings.append(embedding)

faiss.write_index(index, index_path)
print(f"FAISS index saved to {index_path}.")

save_metadata(metadata)

print("Embeddings and index saved successfully.")
