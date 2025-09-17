import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("tested.csv")

# Select useful features and target
df = df[["Sex", "Age", "Fare", "Pclass", "Survived"]].dropna()

# Encode categorical column "Sex"
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# Save model
joblib.dump(model, "time.pkl")
