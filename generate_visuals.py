# generate_visuals_minimal_hybrid.py
"""
Minimal visuals for SQuLMate using full hybrid features:
- Hybrid summarizer confusion matrix
- Q&A question type distribution (bar + table)
- Save summary_labels_generated.csv
- Top sentence importance bar chart (hybrid scores)
- FULL SENTENCE SCORING VISUALIZATION (NEW)
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams.update({'figure.max_open_warning': 0})

# -------------------------
# USER PATHS
# -------------------------
SUMMARIZER_DATASET = r"datasets\DATASET_SQulMate - SUMMARIZER DATASET.csv"
QNA_DATASET = r"C:\Users\HP-USER\Downloads\SQulMate\datasets\DATASET_SQulMate - Q&A DATASET.csv"
HYBRID_MODEL_PATH = r"saved_models/hybrid_summarizer.pkl"
OUT_DIR = "results"
os.makedirs(os.path.join(OUT_DIR, "findings"), exist_ok=True)

# -------------------------
# Load hybrid model
# -------------------------
if not os.path.exists(HYBRID_MODEL_PATH):
    print(f"Hybrid model not found at {HYBRID_MODEL_PATH}. Exiting.")
    sys.exit(1)

HybridModel = joblib.load(HYBRID_MODEL_PATH)
logreg = HybridModel["logreg"]
tfidf_model_saved = HybridModel["tfidf"]
w2v_model = HybridModel["word2vec"]
w2v_dim = HybridModel["feature_info"]["w2v_dim"]

# -------------------------
# Load summarizer dataset
# -------------------------
if not os.path.exists(SUMMARIZER_DATASET):
    print(f"Summarizer dataset not found at {SUMMARIZER_DATASET}. Exiting.")
    sys.exit(1)

df_sum = pd.read_csv(SUMMARIZER_DATASET)
if 'sentence' not in df_sum.columns or 'label' not in df_sum.columns:
    raise ValueError("Summarizer dataset must contain 'sentence' and 'label' columns.")

sentences = df_sum['sentence'].astype(str).tolist()
y_true = df_sum['label'].values

# -------------------------
# Build full hybrid features
# -------------------------

# TF-IDF
tfidf_features = tfidf_model_saved.transform(sentences).toarray()

# Word2Vec
tokenizer = TreebankWordTokenizer()
def get_embedding_for_sentence(s):
    toks = tokenizer.tokenize(str(s).lower())
    vecs = [w2v_model.wv[t] for t in toks if t in w2v_model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(w2v_dim)
embs = np.vstack([get_embedding_for_sentence(s) for s in sentences])

# Advanced Features
def extract_advanced_features(sentences):
    features = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        words = str(sent).split()
        sent_features = [
            i / max(n-1,1),                # Normalized position
            1 if i < 3 else 0,             # Leader
            1 if i >= n-3 else 0,          # Closer
            len(words),                    # Word count
            len(str(sent)),                # Character count
            str(sent).count(","),          # Comma count
            str(sent).count("."),          # Period count
            len(set(words)) / (len(words)+1),  # Lexical diversity
            1 if "important" in str(sent).lower() else 0,
            1 if "conclude" in str(sent).lower() else 0,
            1 if "result" in str(sent).lower() else 0
        ]
        features.append(sent_features)
    return np.array(features)
advanced_feats = extract_advanced_features(sentences)

# TextRank Scores
def textrank_scores_from_tfidf(sentences, tfidf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.transform(sentences)
    sim = (tfidf_matrix * tfidf_matrix.T).toarray()
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank(graph)
    except:
        scores = {i: sim[i].sum() for i in range(len(sentences))}
    return np.array([scores[i] for i in range(len(sentences))])
tr_scores = textrank_scores_from_tfidf(sentences, tfidf_model_saved)

# Combine all features
combined = np.hstack([tfidf_features, embs, advanced_feats, tr_scores.reshape(-1,1)])

# -------------------------
# Predict with hybrid model
# -------------------------
y_pred = logreg.predict(combined)
try:
    y_proba = logreg.predict_proba(combined)[:,1]
except:
    y_proba = np.full(len(y_pred), np.nan)

# Save summary_labels_generated.csv
summary_df = pd.DataFrame({
    "Sentence": sentences,
    "Label": y_true,
    "Hybrid_Score": y_proba
})
summary_csv_path = os.path.join(OUT_DIR, "findings", "summary_labels_generated.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
print("Saved summary_labels_generated.csv ->", summary_csv_path)

# -------------------------
# Hybrid Summarizer Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title("Hybrid Summarizer Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "findings", "hybrid_confusion_matrix.png"))
plt.close(fig)
print("Saved hybrid_confusion_matrix.png")

# -------------------------
# QnA Question Type Distribution
# -------------------------
if os.path.exists(QNA_DATASET):
    df_qna = pd.read_csv(QNA_DATASET)
    if 'questions' in df_qna.columns:
        df_qna['qtype_inferred'] = df_qna['questions'].apply(lambda q: str(q).split()[0].lower() if isinstance(q,str) else "other")
        qtype_counts = df_qna['qtype_inferred'].value_counts()

        # Save bar plot
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(qtype_counts.index.astype(str), qtype_counts.values)
        ax.set_title("Q&A Question Type Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "findings", "qna_question_type_distribution.png"))
        plt.close(fig)
        print("Saved qna_question_type_distribution.png")

        # Save table
        qtype_table_path = os.path.join(OUT_DIR, "findings", "qna_question_type_distribution_table.csv")
        qtype_counts.to_csv(qtype_table_path, header=["Count"])
        print("Saved qna_question_type_distribution_table.csv ->", qtype_table_path)
    else:
        print("QnA dataset missing 'questions' column; skipping visuals.")
else:
    print("QnA dataset not found; skipping visuals.")

# -------------------------
# Top Sentence Importance Bar Chart
# -------------------------
if not np.all(np.isnan(y_proba)):
    top_n = 10
    top_idx = np.argsort(y_proba)[-top_n:][::-1]
    top_sentences = [sentences[i][:80]+"..." if len(sentences[i])>80 else sentences[i] for i in top_idx]
    top_scores = y_proba[top_idx]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(range(len(top_scores)), top_scores, color='skyblue')
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(top_sentences)
    ax.invert_yaxis()
    ax.set_xlabel("Hybrid Summarizer Score")
    ax.set_title(f"Top {top_n} Sentence Importance Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "findings", "top_sentence_scores.png"))
    plt.close(fig)
    print("Saved top_sentence_scores.png")

# ============================================================
# NEW SECTION: FULL SENTENCE-LEVEL SCORING VISUALIZATION
# ============================================================

print("Generating full sentence scoring visualization...")

sentence_scores = {
    "tfidf": tfidf_features.mean(axis=1),
    "word2vec": np.linalg.norm(embs, axis=1),
    "advanced": advanced_feats.mean(axis=1),
    "textrank": tr_scores,
    "hybrid_total": y_proba
}

# BARPLOT OF HYBRID SCORES
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(range(len(sentences)), sentence_scores["hybrid_total"])
ax.set_title("Sentence-Level Hybrid Scoring")
ax.set_xlabel("Sentence Index")
ax.set_ylabel("Hybrid Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "findings", "sentence_hybrid_scoring.png"))
plt.close(fig)
print("Saved sentence_hybrid_scoring.png")

# MULTI-FEATURE CONTRIBUTION STACKED BAR CHART
fig, ax = plt.subplots(figsize=(18, 7))
indices = np.arange(len(sentences))
bottom = np.zeros(len(sentences))

for feature in ["tfidf", "word2vec", "advanced", "textrank"]:
    ax.bar(indices, sentence_scores[feature], bottom=bottom, label=feature)
    bottom += sentence_scores[feature]

ax.set_title("Sentence Feature Contribution Breakdown")
ax.set_xlabel("Sentence Index")
ax.set_ylabel("Score Contribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "findings", "sentence_feature_breakdown.png"))
plt.close(fig)
print("Saved sentence_feature_breakdown.png")


# ============================================================
# TRUE SENTENCE SCORING DEBUG VIEW (REAL RAW SCORES)
# ============================================================

print("Creating sentence_scoring_breakdown.csv ...")

scoring_debug = pd.DataFrame({
    "Sentence": sentences,
    
    # TF-IDF true scoring = sum of tfidf values (sentence importance)
    "TFIDF_Score": tfidf_features.sum(axis=1),
    
    # Word2Vec sentence strength = vector magnitude
    "W2V_Score": np.linalg.norm(embs, axis=1),
    
    # Advanced features = weighted sum of handcrafted features
    "Advanced_Score": advanced_feats.mean(axis=1),
    
    # TextRank pagerank score
    "TextRank_Score": tr_scores,
    
    # Final classifier probability (THIS IS THE HYBRID SCORE)
    "Hybrid_Probability": y_proba
})

debug_path = os.path.join(OUT_DIR, "findings", "sentence_scoring_breakdown.csv")
scoring_debug.to_csv(debug_path, index=False, encoding="utf-8-sig")

print("Saved sentence_scoring_breakdown.csv ->", debug_path)

