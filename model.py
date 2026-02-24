import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Features and target
X = data.drop("condition", axis=1)
y = data["condition"]

# Split dataset and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy*100:.2f}%\n")

# CMD input from user
print("Enter patient details (numbers only):")
user_input = []
for col in X.columns:
    val = float(input(f"{col}: "))
    user_input.append(val)

# Predict
prediction = model.predict([user_input])[0]
if prediction == 1:
    print("\n⚠️ Heart Disease Likely")
else:
    print("\n✅ Heart is Normal")
