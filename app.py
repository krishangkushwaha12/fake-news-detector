from flask import Flask, render_template, request
import pickle
import re

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


model = pickle.load(open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb"))


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", str(text))
    return text.lower()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news = request.form.get("news")
        if news:
            cleaned = clean_text(news)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()

            if prob < 0.8:
                result = "Uncertain ⚠️"
            else:
                result = "Fake News ❌" if prediction == 1 else "Real News ✅"

            confidence = round(prob * 100, 2)

            return render_template(
                "index.html",
                result=result,
                confidence=confidence
            )

    return render_template("index.html")



if __name__ == "__main__":
    app.run
