# =======================================
# qna_train.py
# Train a hybrid Question Type classifier (NER + ML) and test Rule-based templates
# =======================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib
import os

# -------------------------
# 1️⃣ Load Dataset
# -------------------------
dataset_path = r"C:\Users\HP-USER\Downloads\SQulMate\datasets\DATASET_SQulMate - Q&A DATASET.csv"
df = pd.read_csv(dataset_path)

# Ensure columns exist
required_cols = ['source_sentence', 'questions', 'entity']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Dataset must contain column: {col}")

# -------------------------
# 2️⃣ Extract Question Type
# -------------------------
def get_qtype(question):
    first_word = str(question).split()[0].lower()
    if first_word in ["what", "who", "when", "why", "how", "where"]:
        return first_word
    return "what"  # default

df['question_type'] = df['questions'].apply(get_qtype)

# -------------------------
# 3️⃣ Feature Extraction
# -------------------------

# A. TF-IDF vector of source_sentence
tfidf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['source_sentence'])

# B. One-hot encode entity column
encoder = OneHotEncoder(sparse_output=False)
X_entity = encoder.fit_transform(df[['entity']])

# C. Combine features
X = hstack([X_tfidf, X_entity])
y = df['question_type']

# -------------------------
# 4️⃣ Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 5️⃣ Train Logistic Regression
# -------------------------
clf = LogisticRegression(max_iter=400)
clf.fit(X_train, y_train)

# -------------------------
# 6️⃣ Evaluate
# -------------------------
y_pred = clf.predict(X_test)
print("✅ Question Type Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 7️⃣ Save model and resources
# -------------------------
save_path = r"C:\Users\Trisha Mae Cebusana\Downloads\SQulMate\saved_models"
os.makedirs(save_path, exist_ok=True)
joblib.dump({
    "classifier": clf,
    "tfidf": tfidf,
    "entity_encoder": encoder
}, os.path.join(save_path, "qna_classifier.pkl"))
print(f"✅ Model saved at {os.path.join(save_path, 'qna_classifier.pkl')}")

# -------------------------
# 8️⃣ Optional Test: Rule-Based Question Generation
# -------------------------
def generate_question(sentence, q_type):
    if q_type == "what":
        return f"What is {sentence}?"
    elif q_type == "who":
        return f"Who is {sentence}?"
    elif q_type == "when":
        return f"When did {sentence} happen?"
    elif q_type == "why":
        return f"Why {sentence}?"
    elif q_type == "how":
        return f"How {sentence}?"
    elif q_type == "where":
        return f"Where {sentence}?"
    else:
        return f"What is {sentence}?"

# Test on a few sentences from dataset
print("\n--- Sample Generated Questions ---")
for i in range(5):
    sent = df['source_sentence'].iloc[i]
    entity = df['entity'].iloc[i]
    encoded_entity = encoder.transform([[entity]])
    tfidf_vec = tfidf.transform([sent])
    combined_feat = hstack([tfidf_vec, encoded_entity])
    pred_type = clf.predict(combined_feat)[0]
    print(f"Sentence: {sent}")
    print(f"Predicted Question Type: {pred_type}")
    print(f"Generated Question: {generate_question(sent, pred_type)}\n")
