# =======================================
# summarizing_run.py – Hybrid Summarizer
# =======================================

import os
import sys
import re
import numpy as np
import nltk
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
import pandas as pd
from openai import OpenAI

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

# -------------------------
# Get ai_extracted.txt path from argument
# -------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python summarizing_run.py <ai_extracted.txt_path>")
EXTRACTED_SENTENCES_FILE = sys.argv[1]

if not os.path.exists(EXTRACTED_SENTENCES_FILE):
    raise FileNotFoundError(f"{EXTRACTED_SENTENCES_FILE} not found")

# -------------------------
# Load sentences
# -------------------------
sentences = []
with open(EXTRACTED_SENTENCES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() and not line.startswith("="):
            text = line.strip().lstrip("0123456789). -•*").strip()
            sentences.append(text)

# -------------------------
# Load Hybrid Model
# -------------------------
hybrid_model = joblib.load("saved_models/hybrid_summarizer.pkl")
logreg = hybrid_model["logreg"]
tfidf = hybrid_model["tfidf"]
w2v = hybrid_model["word2vec"]

# -------------------------
# Helper functions (embedding, features, textrank, etc.)
# -------------------------
def get_embedding(sentence):
    tokens = word_tokenize(sentence.lower())
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

def extract_advanced_features(sentences):
    features = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        words = sent.split()
        sent_features = [
            i / max(n-1, 1),
            1 if i < 3 else 0,
            1 if i >= n-3 else 0,
            len(words),
            len(sent),
            sent.count(","),
            sent.count("."),
            len(set(words)) / (len(words) + 1),
            1 if "important" in sent.lower() else 0,
            1 if "conclude" in sent.lower() else 0,
            1 if "result" in sent.lower() else 0,
        ]
        features.append(sent_features)
    return np.array(features)

def textrank_scores(sentences):
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    embeddings = np.array([get_embedding(s) for s in sentences])
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    return np.array([scores[i] for i in range(len(sentences))])

def compute_ml_scores(sent_list):
    tfidf_features = tfidf.transform(sent_list).toarray()
    w2v_embeddings = np.array([get_embedding(s) for s in sent_list])
    advanced = extract_advanced_features(sent_list)
    tr = textrank_scores(sent_list).reshape(-1, 1)
    combined = np.hstack([tfidf_features, w2v_embeddings, advanced, tr])
    probs = logreg.predict_proba(combined)[:, 1]
    return probs

# -------------------------
# Score sentences
# -------------------------
ml_scores = compute_ml_scores(sentences)
tr_scores = textrank_scores(sentences)
combined_scores = normalize((ml_scores + tr_scores).reshape(1, -1))[0]

N = max(5, int(len(sentences) * 0.30))
top_indices = combined_scores.argsort()[::-1][:N]

labels = np.zeros(len(sentences), dtype=int)
labels[top_indices] = 1

# -------------------------
# Save CSV
# -------------------------
df = pd.DataFrame({
    "Sentence": sentences,
    "Label": labels,
    "ML_Score": ml_scores,
    "TextRank_Score": tr_scores
})
OUTPUT_CSV = os.path.join(os.path.dirname(EXTRACTED_SENTENCES_FILE), "summary_labels.csv")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# -------------------------
# Build extractive summary (MAINTAIN ORIGINAL ORDER)
# -------------------------
# Sort indices to maintain original document order
top_indices_sorted = sorted(top_indices)

# Extract sentences in their original order
extractive_sentences = [sentences[i] for i in top_indices_sorted]
extractive_text = " ".join(extractive_sentences)

EXTRACTIVE_FILE = os.path.join(os.path.dirname(EXTRACTED_SENTENCES_FILE), "extractive_summary.txt")
with open(EXTRACTIVE_FILE, "w", encoding="utf-8") as f:
    f.write(extractive_text)

# -------------------------
# GitHub AI for three essays
# -------------------------
token = os.getenv("GITHUB_TOKEN")
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
client = OpenAI(api_key=token, base_url=endpoint)

prompt = (
    "Using the extractive text below, generate THREE SEPARATE paragraph:\n\n"
    "1. INTRODUCTION paragraph — A standalone paragraph introducing the topic, providing context, and highlighting importance.\n"
    "2. BODY paragraph — A detailed paragraph explaining key concepts, processes, history, and insights.\n"
    "3. CONCLUSION paragraph — A standalone paragraph summarizing the key points and overall significance.\n\n"
    "Write each essay clearly and cohesively. Do NOT limit length.\n\n"
    + extractive_text
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

three_essays = response.choices[0].message.content
with open(os.path.join(os.path.dirname(EXTRACTED_SENTENCES_FILE), "summary_essays.txt"), "w", encoding="utf-8") as f:
    f.write(three_essays)

# -------------------------
# Print formatted output
# -------------------------
print("\n" + "="*80)
print("EXTRACTIVE SUMMARY:")
print("="*80)
print(extractive_text)
print("\n" + "="*80)
print("ABSTRACTIVE SUMMARY:")
print("="*80)
print(three_essays)
print("\n" + "="*80)
print(f"✅ Files saved:")
print(f"   - Extractive summary: {EXTRACTIVE_FILE}")
print(f"   - Abstractive summary: summary_essays.txt")
print(f"   - Analysis CSV: {OUTPUT_CSV}")
print("="*80)