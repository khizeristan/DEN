# ==============================================================
# Fake News Detector using NLP + Logistic Regression
# ==============================================================
# Dataset: Fake.csv & True.csv (from Kaggle)
# Columns: title, text, label
# ==============================================================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# -------------------------------
# 2. Load Dataset
# -------------------------------
# Replace with your dataset paths
fake = pd.read_csv("D:\Internship-DEN\Project\Fake.csv\Fake.csv")
true = pd.read_csv("D:\Internship-DEN\Project\True.csv\True.csv")

# Add labels
fake["label"] = 0   # Fake = 0
true["label"] = 1   # Real = 1

# Combine datasets
df = pd.concat([fake, true], axis=0).reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------
# 3. Data Preprocessing
# -------------------------------
def clean_text(text):
    """
    Clean the input text:
    - Lowercase
    - Remove URLs, numbers, punctuation
    - Keep only words
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\d+", " ", text)              # remove numbers
    text = re.sub(r"[^a-z\s]", " ", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

# Clean the "text" column
df["clean_text"] = df["text"].apply(clean_text)

# Remove missing/duplicates if any
df.dropna(subset=["clean_text"], inplace=True)
df.drop_duplicates(subset=["clean_text"], inplace=True)

print("After cleaning:", df.shape)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# -------------------------------
# 5. TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# -------------------------------
# 6. Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# -------------------------------
# 7. Evaluation
# -------------------------------
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake","Real"], yticklabels=["Fake","Real"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 9. Save Model + Vectorizer
# -------------------------------
joblib.dump(model, "logreg_tfidf_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved!")

# ==============================================================
# Optional: Streamlit app (save as app.py and run with `streamlit run app.py`)
# ==============================================================

import streamlit as st
import joblib, re

# Load model + vectorizer
model = joblib.load("logreg_tfidf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📰 Fake News Detector")
txt = st.text_area("Paste your news article text here:")

if st.button("Predict"):
    if txt.strip() == "":
        st.warning("Please enter some text!")
    else:
        X = vectorizer.transform([clean_text(txt)])
        proba = model.predict_proba(X)[0,1]
        pred = "Real" if proba >= 0.5 else "Fake"
        conf = proba if pred=="Real" else 1-proba
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {conf:.2%}")

