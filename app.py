from flask import Flask, render_template, request, jsonify
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    prediction = model.predict([message])[0]  # Predict spam (1) or ham (0)
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
