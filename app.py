from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# ==========================
# Load Model and Vectorizer
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "models", "spam_classifier.pkl")

with open(tfidf_path, "rb") as file:
    tfidf = pickle.load(file)

with open(model_path, "rb") as file:
    model = pickle.load(file)


# ==========================
# Input Validation Function
# ==========================

def validate_message(message):

    message = message.strip()

    # Empty input
    if len(message) == 0:
        return False, "Please enter a message."


    # Very long input
    if len(message) > 500:
        return False, "Message is too long. Please enter a smaller SMS."

    clean = message.replace(" ", "")

    # Only numbers
    if clean.isdigit():
        return False, "Please enter a text message instead of only numbers."
    if len(message) < 10:
        return False, "Too short! Please a little longer SMS"
    # Only symbols
    if re.fullmatch(r'[^a-zA-Z0-9]+', clean):
        return False, "Please enter meaningful text."

    # Random alphanumeric garbage
    letters = len(re.findall(r"[A-Za-z]", clean))
    numbers = len(re.findall(r"\d", clean))

    if numbers > letters or letters < 5:
        return False, "Please enter meaningful text."

    # Excessive repeated characters
    if re.search(r"(.)\1{8,}", message):
        return False, "Invalid repeated characters."

    return True, "Valid"


# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/classifier", methods=["POST"])
def classifier():

    msg = request.form["message"]

    # Validate input
    is_valid, validation_message = validate_message(msg)

    if not is_valid:
        return render_template(
            "index.html",
            prediction=validation_message,
            ham_probability="-",
            spam_probability="-"
        )

    # Transform message
    msg_tfidf = tfidf.transform([msg])

    # Predict
    prediction = model.predict(msg_tfidf)[0]

    # Predict probabilities
    probability = model.predict_proba(msg_tfidf)[0]

    ham_probability = probability[0] * 100
    spam_probability = probability[1] * 100

    # Result
    if prediction == 1:
        result = "Spam 🚨"
    else:
        result = "Legitimate ✅"

    return render_template(
        "index.html",
        prediction=result,
        ham_probability=f"{ham_probability:.2f}%",
        spam_probability=f"{spam_probability:.2f}%"
    )


if __name__ == "__main__":
    app.run(debug=True)