# =======================================
# Summarizer Visualization & Multi-Metric Comparison
# Uses saved hybrid model: saved_models/hybrid_summarizer.pkl
# =======================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import TreebankWordTokenizer

# -------------------------------------------------------
# 1️⃣ Load Dataset
# -------------------------------------------------------
dataset_path = r"datasets/DATASET_SQulMate - SUMMARIZER DATASET.csv"
df = pd.read_csv(dataset_path)
X = df["sentence"].astype(str)
y = df["label"]

# -------------------------------------------------------
# 2️⃣ Load Saved Hybrid Model
# -------------------------------------------------------
saved_model_path = "saved_models/hybrid_summarizer.pkl"
HybridModel = joblib.load(saved_model_path)

tfidf = HybridModel["tfidf"]
w2v = HybridModel["word2vec"]

# -------------------------------------------------------
# 3️⃣ TextRank Function
# -------------------------------------------------------
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

def textrank_scores(sentences):
    tfidf_matrix = TfidfVectorizer().fit_transform(sentences)
    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
    np.fill_diagonal(similarity, 0)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)
    return np.array([scores[i] for i in range(len(sentences))]).reshape(-1, 1)

# -------------------------------------------------------
# 4️⃣ Advanced Features
# -------------------------------------------------------
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

# -------------------------------------------------------
# 5️⃣ Word2Vec Embeddings
# -------------------------------------------------------
tokenizer = TreebankWordTokenizer()
def get_embedding(sentence):
    tokens = tokenizer.tokenize(sentence.lower())
    word_vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v.vector_size)

embeddings = np.array([get_embedding(s) for s in X])
advanced = extract_advanced_features(X.tolist())
tfidf_features = tfidf.transform(X).toarray()
textrank = textrank_scores(X.tolist())
combined = np.hstack([tfidf_features, embeddings, advanced, textrank])

# -------------------------------------------------------
# 6️⃣ Train Individual Logistic Regression Models
# -------------------------------------------------------
def train_lr(X_feat, y, desc=""):
    lr = LogisticRegression(max_iter=400, random_state=42)
    lr.fit(X_feat, y)
    print(f"{desc} model trained.")
    return lr

lr_tfidf = train_lr(tfidf_features, y, "TF-IDF Only")
lr_w2v = train_lr(embeddings, y, "Word2Vec Only")
lr_advanced = train_lr(advanced, y, "Advanced Features Only")
lr_textrank = train_lr(textrank, y, "TextRank Only")
lr_combined = HybridModel["logreg"]  # Hybrid already trained

# -------------------------------------------------------
# 7️⃣ Multi-Metric Evaluation
# -------------------------------------------------------
def evaluate_model(X_eval, y_eval, model, name="Model"):
    y_pred = model.predict(X_eval)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_eval, y_pred, average='weighted', zero_division=0),
        "F1-score": f1_score(y_eval, y_pred, average='weighted', zero_division=0)
    }

comparison_results = [
    evaluate_model(tfidf_features, y, lr_tfidf, "TF-IDF Only"),
    evaluate_model(embeddings, y, lr_w2v, "Word2Vec Only"),
    evaluate_model(advanced, y, lr_advanced, "Advanced Features Only"),
    evaluate_model(textrank, y, lr_textrank, "TextRank Only"),
    evaluate_model(combined, y, lr_combined, "Hybrid All Features"),
]

comparison_df = pd.DataFrame(comparison_results)
print("\n📊 Multi-Metric Comparison: Individual vs Combined Features")
print(comparison_df)

# -------------------------------------------------------
# 8️⃣ Visualization
# -------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
comparison_df.set_index("Model", inplace=True)
comparison_df[metrics].plot(kind='bar', figsize=(10,6))
plt.title("Multi-Metric Comparison: Individual vs Combined Features")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
