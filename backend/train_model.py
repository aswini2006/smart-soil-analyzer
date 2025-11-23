# backend/train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "soil_data_small.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "soil_model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def train_and_save(df, model_path=MODEL_PATH):
    X = df[["ph", "moisture", "nitrogen"]]
    y = df["soil_type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    df = load_data()
    train_and_save(df)
