import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
df = pd.read_csv(".csv")  # Make sure this file is in the same folder

# Features and target
X = df[["hours_studied", "sleep_hours", "attendance", "past_score", "distraction_level"]]
y = df["final_score"]

# If distraction_level is categorical, encode it
X = pd.get_dummies(X, columns=["distraction_level"], drop_first=True)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("Model.pkl", "wb") as f:
    pickle.dump(model, f)