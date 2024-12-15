import os
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import fitz
from spacy.lang.en import English
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

data_folder = os.getenv("DATA_FOLDER")
embeddings_folder = os.getenv("EMBEDDINGS_FOLDER")
index_file = os.getenv("INDEX_FILE")

nlp = English()
nlp.add_pipe("sentencizer")

def read_pdfs(folder):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            try:
                with fitz.open(file_path) as pdf:
                    document_pages = [
                        (page.number + 1, page.get_text()) for page in pdf
                    ]
                    documents.append((file, document_pages))
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return documents


def chunk_text_with_metadata(document_pages, file_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_metadata = []
    for page_number, page_text in document_pages:
        chunks = text_splitter.split_text(page_text)
        for chunk_number, chunk in enumerate(chunks, start=1):
            words = len([token for token in nlp(chunk) if token.is_alpha])
            if words > 10:
                chunks_metadata.append(
                    {
                        "file": file_name,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                        "chunk": chunk,
                        "chunk_length": len(chunk)
                    }
                )
    return chunks_metadata


def save_metadata(metadata):
    metadata_path = os.path.join(embeddings_folder, os.getenv("METADATA_FILE"))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}.")


if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

index_path = os.path.join(embeddings_folder, index_file)

documents = read_pdfs(data_folder)

if not documents:
    print("No PDF files found in the data folder.")
    exit()

embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

embedding_dim = int(os.getenv("EMBEDDING_DIM"))
index = faiss.IndexFlatL2(embedding_dim)

metadata = []
all_embeddings = []

for file, document_pages in tqdm(documents, desc="Chunking"):
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
