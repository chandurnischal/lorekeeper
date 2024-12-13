import os
import faiss
import numpy as np
import json
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

embeddings_folder = "embeddings"
index_file = "faiss_index.index"
metadata_file = "metadata.json"

def load_faiss_index():
    index_path = os.path.join(embeddings_folder, index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}. Make sure the embedding script has been run.")
    return faiss.read_index(index_path)

def load_metadata():
    metadata_path = os.path.join(embeddings_folder, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Make sure the embedding script has been run.")
    with open(metadata_path, "r") as f:
        return json.load(f)

def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_embedding = embeddings_model.embed_query(query)

    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)

    relevant_chunks = [metadata[idx] for idx in indices[0]]
    return relevant_chunks, distances[0]

def generate_response(query, relevant_chunks):

    SYSTEM_PROMPT = (
        """
        You are Lorekeeper, an AI assistant specialized in answering questions about *The Lord of the Rings* and *The Hobbit* books. Your knowledge is strictly limited to these books, and you must only use the provided context to generate responses.

        Guidelines:
        1. Only respond using the information contained in the given context.
        2. If the answer to the query is not explicitly found in the context, respond with: "I cannot answer that based on the information provided."
        3. Do not make assumptions, provide opinions, or fabricate information beyond the given context.
        4. Maintain a respectful and neutral tone in your responses.

        Your role is to faithfully assist users in understanding these works while staying true to the source material.  
        """
    )

    context = "\n".join(f"{chunk['chunk']}" for chunk in relevant_chunks)

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\nContext:\n" + context},
            {"role": "user", "content": query},
        ],
    )

    return response["message"]["content"]

def cli():
    try:
        index = load_faiss_index()
        metadata = load_metadata()
    except FileNotFoundError as e:
        print(e)
        return

    print("FAISS index and metadata loaded successfully.")

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        relevant_chunks, _ = retrieve_relevant_chunks(query, index, metadata)
        response = generate_response(query, relevant_chunks)

        print("\nGenerated Response:")
        print(response)