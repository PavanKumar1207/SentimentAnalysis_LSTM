from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
# model_file = "sentiment.pkl"
# with open(model_file, "rb") as file:
#     sentiment_model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        review = request.form.get("review")
        if review:
            prediction = sentiment_model.predict([review])  # Assuming the model uses vectorized text
            result = "Positive" if prediction[0] == 1 else "Negative"

    return render_template("index.html", result='Positive')

if __name__ == "__main__":
    app.run(debug=True)
