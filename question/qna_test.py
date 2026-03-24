# =======================================
# qna_test.py – Q&A generator using Hybrid Summarizer + Hugging Face QG
# =======================================

import os
import re
import joblib
import spacy
import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# NLTK setup
# -------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# -------------------------
# Load Hybrid Summarizer
# -------------------------
model_path = r"C:\Users\HP-USER\Downloads\SQulMate\saved_models\hybrid_summarizer.pkl"
HybridModel = joblib.load(model_path)
logreg_model = HybridModel["logreg"]
tfidf_vectorizer = HybridModel["tfidf"]
w2v_model = HybridModel["word2vec"]

# -------------------------
# Load spaCy for NER
# -------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# -------------------------
# Load Hugging Face QG model
# -------------------------
hf_model_name = "valhalla/t5-small-qg-prepend"
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)

# -------------------------
# Text cleaning
# -------------------------
def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    words = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------
# Word2Vec embedding
# -------------------------
def get_embedding(sentence):
    tokens = tokenizer.tokenize(sentence.lower())
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)

# -------------------------
# Advanced features
# -------------------------
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
            len(set(words)) / (len(words)+1),
            1 if "important" in sent.lower() else 0,
            1 if "conclude" in sent.lower() else 0,
            1 if "result" in sent.lower() else 0
        ]
        features.append(sent_features)
    return np.array(features)

# -------------------------
# TextRank
# -------------------------
def textrank_score(sentences):
    if len(sentences) == 0:
        return np.zeros((0,))
    tfidf_matrix = tfidf_vectorizer.transform(sentences).toarray()
    similarity = tfidf_matrix @ tfidf_matrix.T
    np.fill_diagonal(similarity, 0)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)
    return np.array([scores[i] for i in range(len(sentences))])

# -------------------------
# Improved answer extraction
# -------------------------
def extract_best_answer(sentence):
    doc = nlp(sentence)
    # Prefer NER entities that are meaningful
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "MONEY"]:
            return ent.text
        if ent.label_ in ["DATE", "TIME"] and len(ent.text) > 3:
            return ent.text
    # Otherwise, pick longest noun chunk
    chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    if chunks:
        return max(chunks, key=lambda x: len(x.split()))
    # Fallback to full sentence
    return sentence

# -------------------------
# Hugging Face QG
# -------------------------
def generate_question_hf(sentence, answer):
    input_text = f"answer: {answer} context: {sentence}"
    inputs = hf_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = hf_model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# -------------------------
# Main function
# -------------------------
def main(cleaned_file_path):
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"{cleaned_file_path} not found")

    # Read cleaned sentences
    with open(cleaned_file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Compute importance scores
    scored_sentences = []
    for sent in sentences:
        cleaned = clean_sentence(sent)

        tfidf_feat = tfidf_vectorizer.transform([cleaned]).toarray()
        w2v_feat = get_embedding(cleaned).reshape(1, -1)
        advanced_feat = extract_advanced_features([cleaned])
        textrank_feat = textrank_score([cleaned]).reshape(1, -1)

        combined_feat = np.hstack([tfidf_feat, w2v_feat, advanced_feat, textrank_feat])
        try:
            prob = logreg_model.predict_proba(combined_feat)[0][1]  # probability of being important
        except ValueError:
            continue
        scored_sentences.append((sent, prob))

    # Sort and pick top 10 important sentences
    top_sentences = [s for s, _ in sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:10]]

    # Generate one Q&A per sentence
    qna_pairs = []
    for sent in top_sentences:
        answer = extract_best_answer(sent)
        question = generate_question_hf(sent, answer)
        qna_pairs.append({"question": question, "answer": answer, "source": sent})

    # Print results
    for idx, qa in enumerate(qna_pairs):
        print(f"{idx+1}. Q: {qa['question']}")
        print(f"   A: {qa['answer']}")
        print(f"   Source: {qa['source']}\n")

    print(f"\n✅ Generated {len(qna_pairs)} Q&A pairs.")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    cleaned_file = r"C:\Users\HP-USER\Downloads\SQulMate\ai_extracted.txt"
    main(cleaned_file)
