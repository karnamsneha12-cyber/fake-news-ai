if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, render_template, request, jsonify
import pickle
import requests
import feedparser

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ----------------------------
# AI PREDICT
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = round(max(prob) * 100, 2)
    result = "TRUE ✅" if prediction == 1 else "FAKE ❌"

    return jsonify({
        "result": result,
        "confidence": confidence
    })


# ----------------------------
# BBC NEWS (RSS)
# ----------------------------
@app.route("/bbc")
def bbc_news():
    url = "http://feeds.bbci.co.uk/news/rss.xml"
    feed = feedparser.parse(url)

    news = []
    for entry in feed.entries[:10]:
        news.append({
            "title": entry.title,
            "link": entry.link
        })

    return jsonify(news)


# ----------------------------
# GOOGLE NEWS RSS
# ----------------------------
@app.route("/google")
def google_news():
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    news = []
    for entry in feed.entries[:10]:
        news.append({
            "title": entry.title,
            "link": entry.link
        })

    return jsonify(news)


# ----------------------------
# UI
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)