from flask import Flask,render_template,jsonify,request
import pickle

app = Flask(__name__)

# load models
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


tfidf_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "spam_classifier.pkl"
)


with open(tfidf_path, "rb") as file:
    tfidf = pickle.load(file)


with open(model_path, "rb") as file:
    model = pickle.load(file)
    
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/classifier",methods=["POST"])
def classifier():
    
    msg = request.form['message']
    if len(msg) > 500:
        return render_template(
            "index.html",
            prediction="Message is too long. Please enter a small SMS."
        )

    # transform text
    msg_tfidf = tfidf.transform([msg])
    # prediction
    prediction = model.predict(msg_tfidf)[0]
    
    #probability
    probability = model.predict_proba(msg_tfidf)[0]
    
    spam_probability = probability[1] * 100
    ham_probability = probability[0] * 100


    if prediction == 1:
        result = "Spam 🚨"
    else:
        result = "Legitimate ✅"


    return render_template(
        "index.html",
        prediction=result,
        spam_probability=f"{spam_probability:.2f}%",
        ham_probability=f"{ham_probability:.2f}%"
    )
    
if __name__ == "__main__":
    app.run(debug=True)