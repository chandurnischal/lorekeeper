import os
import faiss
import numpy as np
import json
from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
import ollama

# Directory paths
embeddings_folder = "embeddings"
index_file = "faiss_index.index"
metadata_file = "metadata.json"

app = Flask(__name__)

# Load FAISS index
def load_faiss_index():
    index_path = os.path.join(embeddings_folder, index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}. Make sure the embedding script has been run.")
    return faiss.read_index(index_path)

# Load metadata
def load_metadata():
    metadata_path = os.path.join(embeddings_folder, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Make sure the embedding script has been run.")
    with open(metadata_path, "r") as f:
        return json.load(f)

# Retrieve the most relevant chunks
def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_embedding = embeddings_model.embed_query(query)

    # Search in FAISS index
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)

    # Extract the relevant chunks from metadata
    relevant_chunks = [metadata[idx] for idx in indices[0]]
    return relevant_chunks, distances[0]

# Generate response using the Llama model
def generate_response(query, relevant_chunks):
    SYSTEM_PROMPT = (
        """You are a helpful reading assistant who answers questions "
        "based on snippets of text provided in context. Answer only using the context provided, "
        "being as concise as possible. If you're unsure, just say that you don't know. "
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

# API endpoint to handle queries
@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required."})

        index = load_faiss_index()
        metadata = load_metadata()

        relevant_chunks, _ = retrieve_relevant_chunks(query, index, metadata)
        response = generate_response(query, relevant_chunks)

        return jsonify({"response": response}), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)