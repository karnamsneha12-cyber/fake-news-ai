# VeriNews — AI Fake News Detector

A modern, fully responsive Fake News Detection website built with **Flask + scikit-learn + NLTK**.
Uses **TF-IDF vectorization** and **Multinomial Naive Bayes** to classify news as **REAL** or **FAKE**
with a confidence score, wrapped in a sleek animated UI.

## 📁 Project structure

```
fake-news-detector/
├── app.py               # Flask backend
├── train_model.py       # Trains and saves model.pkl + vectorizer.pkl
├── requirements.txt
├── Fake.csv             # ← download from Kaggle
├── True.csv             # ← download from Kaggle
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## 🚀 How to run locally

### 1. Get the datasets
Download the **Fake and Real News Dataset** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` in the project root.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (only once)
```bash
python train_model.py
```
This creates `model.pkl` and `vectorizer.pkl`. Expect ~95% accuracy.

### 4. Run the website
```bash
python app.py
```
Open http://localhost:5000 in your browser. 🎉

## 🧠 How it works
1. Text is cleaned with NLTK (lowercase, remove URLs/punctuation/stopwords, Porter stemming).
2. TF-IDF turns cleaned text into a numerical feature vector (5,000 features, 1–2 grams).
3. A Multinomial Naive Bayes classifier predicts REAL (1) or FAKE (0) and outputs a confidence %.
4. Flask serves the static frontend and the `/predict` JSON endpoint.

## 🎨 UI features
- Glassmorphism + gradient design
- Sticky blurred navbar
- Hero with animated stats
- Loading spinner during prediction
- Color-coded result card (green = REAL, red = FAKE) with animated confidence bar
- Fully responsive (mobile, tablet, desktop)

Perfect for college submissions and resume projects. ✨
