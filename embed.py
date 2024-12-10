import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
from tqdm import tqdm

def process_pdfs_to_faiss(data_dir, faiss_index_file, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    docs = []
    for file_name in tqdm(os.listdir(data_dir)):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file_name))
            raw_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(raw_documents)
            for doc in documents:
                doc.metadata["source"] = file_name
            docs.extend(documents)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    with open(faiss_index_file, "wb") as f:
        pickle.dump(vectorstore, f)
    
    print(f"FAISS index saved to {faiss_index_file}")

if __name__ == "__main__":
    data_dir = "data" 
    faiss_index_file = "database/faiss_index.pkl"
    process_pdfs_to_faiss(data_dir, faiss_index_file)
