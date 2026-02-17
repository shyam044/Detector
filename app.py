import re
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder="templates")

# Load the model and tokenizer
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Lightweight safer preprocessing (no NLTK, no WordNet)
def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)     # remove punctuation
    return text.split()

def predict_spam(email_text):
    original_text = email_text.lower()

    # --- STAGE 1: REGEX SCAM SIGNATURES ---
    if re.search(r'http[s]?://\S+', original_text):
        return "Spam (Suspicious Link Detected)"

    scam_patterns = [
        'unpaid fee',
        'pending delivery',
        'action required',
        'click here',
        'package delivery'
    ]

    if any(p in original_text for p in scam_patterns) and len(original_text) < 200:
        return "Spam (SMS Phishing Pattern)"

    # --- STAGE 2: MODEL CLASSIFICATION ---
    words = simple_preprocess(original_text)
    cleaned = " ".join(words)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=150, padding='post')
    prediction_score = model.predict(padded)[0][0]

    # Dynamic confidence threshold
    threshold = 0.85 if len(original_text) > 250 else 0.45

    if prediction_score > threshold:
        return "Spam"
    return "Ham"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        result = predict_spam(email_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
