# =======================================
# Unified Hybrid Lecture Summarizer Trainer
# (TF-IDF + Word2Vec + Advanced Features + Logistic Regression + TextRank)
# Outputs ONE single model file.
# =======================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from nltk.tokenize import TreebankWordTokenizer
import os, joblib
import sys
import networkx as nx

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------
# 1️⃣ Load Dataset
# -------------------------------------------------------
dataset_path = r"datasets/DATASET_SQulMate - SUMMARIZER DATASET.csv"
df = pd.read_csv(dataset_path)
X = df["sentence"].astype(str)  # Convert to string to avoid NaNs
y = df["label"]

# -------------------------------------------------------
# 2️⃣ TextRank function (for hybrid scoring)
# -------------------------------------------------------
def textrank_scores(sentences):
    tfidf_matrix = TfidfVectorizer().fit_transform(sentences)
    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
    np.fill_diagonal(similarity, 0)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)
    return np.array([scores[i] for i in range(len(sentences))])

# -------------------------------------------------------
# 3️⃣ Advanced handcrafted features
# -------------------------------------------------------
def extract_advanced_features(sentences):
    features = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        words = sent.split()
        sent_features = [
            i / max(n-1, 1),                        # normalized position
            1 if i < 3 else 0,                       # beginning of doc
            1 if i >= n-3 else 0,                    # end of doc
            len(words),                               # word count
            len(sent),                                # char count
            sent.count(","),                          # commas
            sent.count("."),                          # periods
            len(set(words)) / (len(words) + 1),      # unique word ratio
            1 if "important" in sent.lower() else 0,
            1 if "conclude" in sent.lower() else 0,
            1 if "result" in sent.lower() else 0,
        ]
        features.append(sent_features)
    return np.array(features)

# -------------------------------------------------------
# 4️⃣ Train Word2Vec (using TreebankWordTokenizer)
# -------------------------------------------------------
tokenizer = TreebankWordTokenizer()
tokenized = [tokenizer.tokenize(s.lower()) for s in X]

w2v = Word2Vec(
    tokenized,
    vector_size=150,
    window=5,
    min_count=1,
    workers=os.cpu_count(),
    seed=42
)
w2v.train(tokenized, total_examples=len(tokenized), epochs=12)

def get_embedding(sentence):
    tokens = tokenizer.tokenize(sentence.lower())
    word_vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v.vector_size)

# -------------------------------------------------------
# 5️⃣ TF-IDF Features
# -------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)
tfidf_features = tfidf.fit_transform(X)

# -------------------------------------------------------
# 6️⃣ TextRank Scores
# -------------------------------------------------------
textrank = textrank_scores(X.tolist()).reshape(-1, 1)

# -------------------------------------------------------
# 7️⃣ Combine ALL Features
# -------------------------------------------------------
embeddings = np.array([get_embedding(s) for s in X])
advanced = extract_advanced_features(X.tolist())
combined = np.hstack([
    tfidf_features.toarray(),
    embeddings,
    advanced,
    textrank
])

# -------------------------------------------------------
# 8️⃣ Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    combined, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# 9️⃣ Train Logistic Regression (GridSearch)
# -------------------------------------------------------
params = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=400, random_state=42),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

y_pred = model.predict(X_test)
print("Best parameters:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------------
# 🔥 10️⃣ Save EVERYTHING AS ONE SINGLE MODEL
# -------------------------------------------------------
HybridModel = {
    "logreg": model,
    "tfidf": tfidf,
    "word2vec": w2v,
    "feature_info": {
        "w2v_dim": w2v.vector_size,
        "advanced_feature_dim": advanced.shape[1],
        "textrank_included": True
    }
}

os.makedirs("saved_models", exist_ok=True)
joblib.dump(HybridModel, "saved_models/hybrid_summarizer.pkl")

print("\n🎉 ONE unified hybrid model saved as: saved_models/hybrid_summarizer.pkl")
