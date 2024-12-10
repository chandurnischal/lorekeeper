import pickle
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()

def query_faiss_index(faiss_index_file, query, model_name="meta-llama/Llama-3.2-1B"):

    with open(faiss_index_file, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_KEY"))
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_KEY"))
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=200
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa_chain = load_qa_chain(llm, chain_type="stuff")

    retrieval_qa_chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever, return_source_documents=True)

    result = retrieval_qa_chain({"query": query})
    answer = result["result"]
    source_docs = result["source_documents"]

    print(f"Answer: {answer}")
    print("\nSources:")
    for doc in source_docs:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
    
    return answer

if __name__ == "__main__":
    
    query = input("Enter your query: ")
    faiss_index_file = "database/faiss_index.pkl"
    query_faiss_index(faiss_index_file, query)
