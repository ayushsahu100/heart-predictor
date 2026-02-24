from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset safely (important for Render Linux environment)
data_path = os.path.join(os.path.dirname(__file__), "heart.csv")
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
        user_input = [float(request.form[col]) for col in X.columns]
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            result = "Heart Disease Likely ⚠️"
        else:
            result = "No Heart Disease ✅"

    except Exception as e:
        result = "Error: Please enter valid numbers for all fields"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run()
