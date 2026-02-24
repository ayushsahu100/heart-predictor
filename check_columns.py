


import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load dataset safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder
data_path = os.path.join(BASE_DIR, "heart.csv")
data = pd.read_csv(data_path)

X = data.drop("condition", axis=1)
y = data["condition"]
model = RandomForestClassifier()
model.fit(X, y)
