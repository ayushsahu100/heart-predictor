from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- Load dataset safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder
data_path = os.path.join(BASE_DIR, "heart.csv")
data = pd.read_csv(data_path)

X = data.drop("condition", axis=1)
y = data["condition"]
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        user_input = [float(request.form[col]) for col in X.columns]
        prediction = model.predict([user_input])[0]
        result = "⚠️ Heart Disease Likely" if prediction == 1 else "✅ Heart Normal"
    except:
        result = "Error: Please enter valid numbers for all fields"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
