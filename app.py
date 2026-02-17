import re
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import nltk
nltk.data.path.append("./nltk_data")  # must be BEFORE WordNetLemmatizer import

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()

def predict_spam(email_text):
    original_text = email_text.lower()
    
    # --- STAGE 1: REGEX SCAM SIGNATURES (High Accuracy for SMS) ---
    # Catching suspicious links (USPS/Phishing)
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', original_text):
        return "Spam (Suspicious Link Detected)"
    
    # Catching "Urgency + Fee" patterns common in USPS scams
    scam_patterns = ['unpaid fee', 'pending delivery', 'action required', 'click here', 'package delivery']
    if any(pattern in original_text for pattern in scam_patterns) and len(original_text) < 200:
        return "Spam (SMS Phishing Pattern)"

    # --- STAGE 2: AI MODEL WITH DYNAMIC THRESHOLD ---
    # Clean text (matching your training script)
    words = [lemmatizer.lemmatize(w) for w in original_text.split()]
    cleaned = " ".join(words)
    
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=150, padding='post')
    prediction_score = model.predict(padded)[0][0]

    # Rule: If it's a long message (News), we need 85% certainty to call it spam.
    # If it's short (SMS), we only need 45% certainty.
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
