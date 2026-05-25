"""
train_model.py
---------------
Train Fake News Detection Model
"""

import re
import pickle
import string

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# NLTK
# -----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned_words = []

    for word in words:

        if word not in STOPWORDS and len(word) > 2:

            cleaned_words.append(
                LEMMATIZER.lemmatize(word)
            )

    return " ".join(cleaned_words)


# -----------------------------
# LOAD DATASET
# -----------------------------
print("📥 Loading datasets...")

fake = pd.read_csv(
    "Fake.csv",
    low_memory=False
)

true = pd.read_csv(
    "True.csv",
    low_memory=False
)

# IMPORTANT
# 0 = FAKE
# 1 = REAL

fake["label"] = 0

true["label"] = 1

df = pd.concat(
    [fake, true],
    ignore_index=True
)

df = df[["text", "label"]]

df.dropna(inplace=True)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("🧹 Cleaning text...")

df["clean"] = df["text"].apply(clean_text)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(

    df["clean"],

    df["label"],

    test_size=0.2,

    random_state=42

)


# -----------------------------
# TFIDF
# -----------------------------
print("🔠 Vectorizing...")

vectorizer = TfidfVectorizer(

    max_features=30000,

    ngram_range=(1, 2)

)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# MODEL
# -----------------------------
print("🤖 Training AI Model...")

model = LogisticRegression(

    max_iter=5000

)

model.fit(

    X_train_vec,

    y_train

)


# -----------------------------
# EVALUATION
# -----------------------------
preds = model.predict(X_test_vec)

accuracy = accuracy_score(

    y_test,

    preds

)

print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%\n")

print(

    classification_report(

        y_test,

        preds,

        target_names=[

            "FAKE",

            "REAL"

        ]

    )

)


# -----------------------------
# SAVE
# -----------------------------
print("💾 Saving model...")

with open("model.pkl", "wb") as f:

    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:

    pickle.dump(vectorizer, f)

print("✅ Training Completed")