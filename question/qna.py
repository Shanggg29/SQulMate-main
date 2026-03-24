# =======================================
# qna.py – Q&A generator using Hybrid Summarizer + Hugging Face QG
# Improved version for better Q&A quality
# =======================================

import os
import sys
import re
import joblib
import spacy
import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
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
    nlp = spacy.load("en_core_web_trf")  # transformer-based model for better accuracy
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_trf"], check=True)
    nlp = spacy.load("en_core_web_trf")

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
# Improved Answer extraction
# -------------------------
def extract_best_answer(sentence):
    doc = nlp(sentence)
    
    # Prefer named entities with length > 1
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "MONEY", "DATE", "TIME"]:
            if len(ent.text.strip()) > 1:
                return ent.text.strip()
    
    # If no entities, pick the most "central" noun chunk based on TF-IDF weight
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    if noun_chunks:
        # Compute TF-IDF weight for each chunk
        chunk_weights = []
        for chunk in noun_chunks:
            chunk_vec = tfidf_vectorizer.transform([chunk]).toarray().sum()
            chunk_weights.append(chunk_vec)
        return noun_chunks[np.argmax(chunk_weights)]
    
    # fallback to sentence if nothing else
    return sentence.strip()

# -------------------------
# Improved Hugging Face QG
# -------------------------
def generate_question_hf(sentence, answer):
    if not answer or len(answer.split()) < 1:
        return ""  # skip if answer is empty
    input_text = f"answer: {answer} context: {sentence}"
    inputs = hf_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = hf_model.generate(inputs, max_length=64, num_beams=6, early_stopping=True)
    question = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Filter poor questions
    if len(question.split()) < 3 or len(question.split()) > 25:
        return ""
    return question

# -------------------------
# Main function
# -------------------------
def main(ai_extracted_path):
    if not os.path.exists(ai_extracted_path):
        print(f"{ai_extracted_path} not found")
        return

    with open(ai_extracted_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    scored_sentences = []
    for sent in sentences:
        cleaned = clean_sentence(sent)
        tfidf_feat = tfidf_vectorizer.transform([cleaned]).toarray()
        w2v_feat = get_embedding(cleaned).reshape(1, -1)
        advanced_feat = extract_advanced_features([cleaned])
        textrank_feat = textrank_score([cleaned]).reshape(1, -1)
        combined_feat = np.hstack([tfidf_feat, w2v_feat, advanced_feat, textrank_feat])
        try:
            prob = logreg_model.predict_proba(combined_feat)[0][1]
        except ValueError:
            continue
        scored_sentences.append((sent, prob))

    top_sentences = [s for s, _ in sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:10]]

    qna_pairs = []
    for sent in top_sentences:
        answer = extract_best_answer(sent)
        question = generate_question_hf(sent, answer)
        if question:  # skip empty or poor questions
            qna_pairs.append(f"Q: {question}\nA: {answer}\nSource: {sent}\n")

    output_path = os.path.join(os.getcwd(), "uploads", "qa_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(qna_pairs))

    print(f"✅ Generated {len(qna_pairs)} Q&A pairs at {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qna.py <path_to_ai_extracted.txt>")
    else:
        main(sys.argv[1])
