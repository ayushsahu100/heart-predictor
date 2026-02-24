import pandas as pd

data = pd.read_csv("heart.csv")
print("Columns in CSV:")
for col in data.columns:
    print(col)
