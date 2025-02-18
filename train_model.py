import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("spam.csv")
df.columns = df.columns.str.strip()  # Remove spaces from column names

print("✅ Dataset Loaded Successfully!")
print("First few rows:\n", df.head())
print("Total Rows before cleaning:", df.shape[0])

# Check for missing values before cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop rows where "label" is missing
df.dropna(subset=["label"], inplace=True)

# Fill missing "message" values with a placeholder instead of dropping them
df["message"] = df["message"].fillna("EMPTY_MESSAGE")

# Convert labels to binary (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Remove rows where label conversion failed (if any)
df.dropna(subset=["label"], inplace=True)

# Ensure dataset has enough rows
print("Total Rows after cleaning:", df.shape[0])
if df.shape[0] < 10:
    raise ValueError(f"Dataset too small! Found only {df.shape[0]} rows. Add more samples.")

# Check label distribution
print("Label distribution:\n", df["label"].value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough data for training/testing. Increase dataset size.")

# Create a text-processing and classification pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete. Saved as model.pkl.")
