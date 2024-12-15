from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import torch
from transformers import BertTokenizer, BertModel
from collections import Counter

import nltk

nltk.download("stopwords")
nltk.download("punkt_tab")


# Function to load text from a file
def load_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Preprocess text by tokenizing and removing stopwords/punctuation
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)


# Cosine Similarity using TF-IDF
def cosine_similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]


# BERT-based Semantic Similarity
def bert_similarity(text1, text2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    def get_bert_embedding(text):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    emb1 = get_bert_embedding(text1)
    emb2 = get_bert_embedding(text2)

    # Cosine similarity between BERT embeddings
    similarity = cosine_similarity([emb1], [emb2])
    return similarity[0][0]


# Jaccard Similarity
def jaccard_similarity(text1, text2):
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


# Main function to compare two text files
def compare_texts(file1, file2):
    text1 = load_text(file1)
    text2 = load_text(file2)

    print("Preprocessing texts...")
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Cosine Similarity using TF-IDF
    tfidf_similarity = cosine_similarity_tfidf(text1, text2)

    # BERT-based Semantic Similarity
    bert_sim = bert_similarity(text1, text2)

    # Jaccard Similarity
    jaccard_sim = jaccard_similarity(text1, text2)

    print("Overall Score: ", round((tfidf_similarity + bert_sim + jaccard_sim) / 3, 3))


file1 = "reference.txt"
file2 = "response.txt"
compare_texts(file1, file2)
