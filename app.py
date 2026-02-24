from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

model = None
X_columns = None

def load_model():
    global model, X_columns
    
    data_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    data = pd.read_csv(data_path)

    X = data.drop("condition", axis=1)
    y = data["condition"]

    X_columns = X.columns.tolist()

    model = RandomForestClassifier()
    model.fit(X, y)

load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = [float(request.form[col]) for col in X_columns]
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            result = "Heart Disease Likely ⚠️"
        else:
            result = "No Heart Disease ✅"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
