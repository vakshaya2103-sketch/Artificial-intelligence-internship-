"""
Project 2: Data Classification Using AI
========================================

Goal:
    Build a basic classification model using a small dataset.

Key Requirements covered:
    1. Load and understand a dataset
    2. Split data into training and testing sets
    3. Apply a simple classification algorithm

Key Skills demonstrated:
    Data handling, supervised learning basics, model training.

Dataset:
    This project uses the classic Iris flower dataset (built into
    scikit-learn, so no download or internet connection is needed).
    It contains 150 flower samples, 4 numeric features each (petal/sepal
    length & width), and a target label (the flower species).
    It's the standard "hello world" dataset for classification because
    it's small, clean, and easy to reason about.

    To use your OWN dataset instead, see the `load_custom_csv()` function
    near the bottom and the note in `main()`.

Author: (Your Name Here)
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# 1. LOAD AND UNDERSTAND THE DATASET
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """
    Loads the Iris dataset and returns it as a single pandas DataFrame
    with feature columns plus a 'species' label column.

    Understanding the dataset is a key requirement, so this function
    also prints a quick summary: shape, first few rows, and class
    distribution.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # The raw target column is named "target" and holds integers (0, 1, 2).
    # We map it to readable species names for clarity.
    df["species"] = df["target"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    df = df.drop(columns=["target"])

    print("=" * 60)
    print(" STEP 1: LOAD & UNDERSTAND THE DATASET")
    print("=" * 60)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Feature summary statistics:")
    print(df.describe(), "\n")
    print("Class distribution (target balance check):")
    print(df["species"].value_counts(), "\n")

    return df


# ---------------------------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets.

    - X = features (everything except the label column)
    - y = target/label column ('species')

    test_size=0.2 means 80% of the data is used to train the model and
    20% is held back to evaluate it on data it has never seen.
    random_state is fixed so the split is reproducible every time you run
    this script (same train/test rows each run).
    """
    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("=" * 60)
    print(" STEP 2: TRAIN/TEST SPLIT")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. APPLY A SIMPLE CLASSIFICATION ALGORITHM
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    """
    Trains a K-Nearest Neighbors (KNN) classifier — a simple, intuitive
    algorithm that classifies a new sample by looking at the 'k' most
    similar samples in the training data and taking a majority vote.

    Steps:
      1. Scale features (KNN is distance-based, so features on different
         scales would unfairly dominate the distance calculation).
      2. Fit the classifier on the scaled training data.
    """
    print("=" * 60)
    print(" STEP 3: TRAIN THE CLASSIFICATION MODEL")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    print("Model: K-Nearest Neighbors (k=5)")
    print("Training complete.\n")

    return model, scaler


# ---------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# ---------------------------------------------------------------------------
def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluates the trained model on the held-out test set and prints:
      - Overall accuracy
      - A full classification report (precision/recall/F1 per class)
      - A confusion matrix (which classes get mixed up, if any)
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(" STEP 4: EVALUATE ON TEST DATA")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    return y_pred


# ---------------------------------------------------------------------------
# 5. PREDICT ON NEW / UNSEEN DATA
# ---------------------------------------------------------------------------
def predict_new_sample(model, scaler, sample: list) -> str:
    """
    Demonstrates using the trained model to classify a brand-new,
    unseen flower measurement.

    sample = [sepal_length, sepal_width, petal_length, petal_width]
    """
    sample_df = pd.DataFrame([sample], columns=scaler.feature_names_in_)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)[0]
    return prediction


# ---------------------------------------------------------------------------
# OPTIONAL: LOAD YOUR OWN CSV INSTEAD OF THE BUILT-IN DATASET
# ---------------------------------------------------------------------------
def load_custom_csv(path: str, label_column: str) -> pd.DataFrame:
    """
    Loads a custom dataset from a CSV file instead of the built-in Iris
    dataset. Use this if your internship wants you to classify your own
    data.

    path:          path to your .csv file
    label_column:  the name of the column you want to predict (the target)

    Example:
        df = load_custom_csv("my_data.csv", label_column="species")
    """
    df = pd.read_csv(path)
    df = df.rename(columns={label_column: "species"})
    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    # --- Step 1: Load and understand the dataset ---
    df = load_dataset()
    # To use your own data instead, comment the line above and uncomment:
    # df = load_custom_csv("my_data.csv", label_column="species")

    # --- Step 2: Split into training and testing sets ---
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Step 3: Train a simple classification model ---
    model, scaler = train_model(X_train, y_train)

    # --- Step 4: Evaluate the model on unseen test data ---
    evaluate_model(model, scaler, X_test, y_test)

    # --- Step 5: Try it on one brand-new sample ---
    print("=" * 60)
    print(" STEP 5: PREDICT ON A NEW SAMPLE")
    print("=" * 60)
    new_sample = [5.1, 3.5, 1.4, 0.2]  # looks like a setosa flower
    result = predict_new_sample(model, scaler, new_sample)
    print(f"Input measurements: {new_sample}")
    print(f"Predicted species: {result}")


if __name__ == "__main__":
    main()
